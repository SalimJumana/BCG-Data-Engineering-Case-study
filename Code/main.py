from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *

import os
import sys

if os.path.exists('src.zip'):
    sys.path.insert(0, 'src.zip')
else:
    sys.path.insert(0, './Code/src')

from src.utilities import utils


class CarCrash_Spark_Application:
    def __init__(self, path_to_config_file):
        input_file_paths = utils.read_yaml(path_to_config_file).get("INPUT_FILENAME")
        self.df_charges = utils.load_csv_data_to_df(spark, input_file_paths.get("Charges"))
        self.df_damages = utils.load_csv_data_to_df(spark, input_file_paths.get("Damages"))
        self.df_endorse = utils.load_csv_data_to_df(spark, input_file_paths.get("Endorse"))
        self.df_primperson = utils.load_csv_data_to_df(spark, input_file_paths.get("Primary_Person"))
        self.df_units = utils.load_csv_data_to_df(spark, input_file_paths.get("Units"))
        self.df_restrict = utils.load_csv_data_to_df(spark, input_file_paths.get("Restrict"))

    def count_crash_male_more_than_2(self, output_path, output_format):
        """
        Find the number of crashes (accidents) in which number of males killed are greater than 2?
        :param output_path: output file path
        :param output_format: Write file format
        :return: count of crashes
        """
        df = self.df_primperson.filter((col('PRSN_GNDR_ID') == 'MALE') & (col('DEATH_CNT') == 1)). \
            groupBy('CRASH_ID').agg(sum(col("DEATH_CNT")).alias("NO_OF_MALES_KILLED")). \
            filter(col("NO_OF_MALES_KILLED") > 2)

        utils.write_output(df, output_path, output_format)

        return df.first().NO_OF_MALES_KILLED

    def count_2_wheeler_accidents(self, output_path, output_format):
        """
        How many two wheelers are booked for crashes?
        :param output_format: Write file format
        :param output_path: output file path
        :return: count of two wheelers
        """
        df_two_wheelers = self.df_units.filter(col('VEH_BODY_STYL_ID').contains("MOTORCYCLE")). \
            select(countDistinct('VIN').alias('NO_OF_DISTINCT_TWO_WHEELERS'))

        utils.write_output(df_two_wheelers, output_path, output_format)

        return df_two_wheelers.first()

    def get_top_vehmake_driver_died_airbags_notdeployed(self, output_path, output_format):
        """
        Determine the Top 5 Vehicle Makes of the cars present in the crashes in which driver died and Airbags did not deploy
        :param output_format: Write file format
        :param output_path: output file path
        :return: list of vehicle make IDs
        """
        df = self.df_primperson.alias("pp").join(self.df_units, on=['CRASH_ID', 'UNIT_NBR'], how="inner"). \
            filter((col('PRSN_TYPE_ID').contains('DRIVER')) & (col('PRSN_AIRBAG_ID') == 'NOT DEPLOYED') & \
            (col('pp.DEATH_CNT') == 1)). \
            filter(col('VEH_BODY_STYL_ID').contains('CAR')). \
            groupBy('VEH_MAKE_ID').agg(countDistinct("VIN").alias('COUNT_OF_VEHICLES_PER_VEH_MAKE')). \
            orderBy(col('COUNT_OF_VEHICLES_PER_VEH_MAKE').desc(), col('VEH_MAKE_ID').asc()).limit(5)

        utils.write_output(df, output_path, output_format)

        return [veh[0] for veh in df.select("VEH_MAKE_ID").collect()]

    def get_no_vehicles_with_license_hit_run(self, output_path, output_format):
        """
        Determine number of Vehicles with driver having valid licences involved in hit and run?
        :param output_format: Write file format
        :param output_path: output file path
        :return: No of vehicles count
        """
        df_no_of_vehicles = self.df_primperson.join(self.df_units, on=['CRASH_ID', 'UNIT_NBR'], how="inner"). \
            filter((col('PRSN_TYPE_ID').contains('DRIVER')) & (col('DRVR_LIC_CLS_ID').contains('CLASS'))). \
            filter(col('VEH_HNR_FL') == 'Y'). \
            select(countDistinct("VIN").alias("NO_OF_VEHICLES"))

        utils.write_output(df_no_of_vehicles, output_path, output_format)

        return df_no_of_vehicles.first()

    def get_state_highest_accident_no_female(self, output_path, output_format):
        """
        Which state has highest number of accidents in which females are not involved?
        :param output_format: Write file format
        :param output_path: output file path
        :return: state with highest accidents with no female
        """
        df_crash_with_no_female = self.df_primperson.\
            withColumn('NO_OF_FEMALES', when(col('PRSN_GNDR_ID') == 'FEMALE', 1).otherwise(0)). \
            groupBy(col('CRASH_ID')).agg(sum(col('NO_OF_FEMALES')).alias('NO_OF_FEMALES_PER_CRASH')). \
            filter(col('NO_OF_FEMALES_PER_CRASH') == 0).select('CRASH_ID')

        df_state_most_accidents = df_crash_with_no_female.join(self.df_primperson, on='CRASH_ID', how='inner'). \
            groupBy('DRVR_LIC_STATE_ID').count(). \
            filter(~col("DRVR_LIC_STATE_ID").isin("Other", "Unknown", "NA")). \
            orderBy(col("count").desc()).limit(1)

        utils.write_output(df_state_most_accidents, output_path, output_format)

        return df_state_most_accidents.first().DRVR_LIC_STATE_ID

    def get_top_vehicle_make_max_injuries(self, output_path, output_format):
        """
        Which are the Top 3rd to 5th VEH_MAKE_IDs that contribute to a largest number of injuries including death
        :param output_format: Write file format
        :param output_path: output file path
        :return: list of vehicle make IDs
        """
        w = Window.orderBy(col("TOTAL_INJURIES_VEH_MAKEID").desc())

        df_3to5_veh_make_id = self.df_units.withColumn('TOTAL_INJURIES', col('TOT_INJRY_CNT') + col('DEATH_CNT')). \
            groupby("VEH_MAKE_ID").agg(sum(col("TOTAL_INJURIES")).alias('TOTAL_INJURIES_VEH_MAKEID')). \
            withColumn("row", row_number().over(w)).filter(col("row").isin([3, 4, 5]))

        utils.write_output(df_3to5_veh_make_id, output_path, output_format)

        return [veh[0] for veh in df_3to5_veh_make_id.select("VEH_MAKE_ID").collect()]

    def get_top_ethnic_ug_each_body_style(self, output_path, output_format):
        """
        For all the body styles involved in crashes, mention the top ethnic user group of each unique body style
        :param output_format: Write file format
        :param output_path: output file path
        :return: None
        """
        w = Window.partitionBy("VEH_BODY_STYL_ID").orderBy(col("count").desc())

        df = self.df_primperson.alias('pp').join(self.df_units.alias('u'), on=['CRASH_ID', 'UNIT_NBR'], how='inner'). \
            filter(~col('u.VEH_BODY_STYL_ID').isin(["NA", "UNKNOWN", "NOT REPORTED", "OTHER  (EXPLAIN IN NARRATIVE)"])). \
            filter(~col('pp.PRSN_ETHNICITY_ID').isin(["NA", "UNKNOWN"])). \
            groupby("VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID").count(). \
            withColumn("row", row_number().over(w)).filter(col("row") == 1).drop("row", "count")

        utils.write_output(df, output_path, output_format)

        df.show(truncate=False)

    def get_top_5_zip_codes_crash_with_alcohol(self, output_path, output_format):
        """
        Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols
        as the contributing factor to a crash (Use Driver Zip Code)
        :param output_format: Write file format
        :param output_path: output file path
        :return: List of Zip Codes
        """
        df = self.df_units.join(self.df_primperson, on=['CRASH_ID'], how='inner'). \
            dropna(subset=["DRVR_ZIP"]). \
            filter(col("CONTRIB_FACTR_1_ID").contains("ALCOHOL") | col("CONTRIB_FACTR_2_ID").contains("ALCOHOL")). \
            groupby("DRVR_ZIP").count().orderBy(col("count").desc()).limit(5).drop("count")

        utils.write_output(df, output_path, output_format)

        return [row[0] for row in df.collect()]

    def get_crash_ids_with_no_damage(self, output_path, output_format):
        """
        Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is
        above 4 and car avails Insurance
        :param output_format: Write file format
        :param output_path: output file path
        :return: None
        """
        df = self.df_damages.alias('d').join(self.df_units.alias('u'), on=["CRASH_ID"], how='inner'). \
            filter(
            ((col('u.VEH_DMAG_SCL_1_ID') > "DAMAGED 4") & (
                ~col('u.VEH_DMAG_SCL_1_ID').isin(["NA", "NO DAMAGE", "INVALID VALUE"]))
             ) |
            ((col('u.VEH_DMAG_SCL_2_ID') > "DAMAGED 4") & (
                ~col('u.VEH_DMAG_SCL_2_ID').isin(["NA", "NO DAMAGE", "INVALID VALUE"]))
             )). \
            filter(col('d.DAMAGED_PROPERTY') == "NONE"). \
            filter(col('u.FIN_RESP_TYPE_ID') != "NA"). \
            select(countDistinct('u.CRASH_ID').alias("COUNT_OF_DISTINCT_CRASH_IDS"))

        utils.write_output(df, output_path, output_format)

        df.show()

    def get_top_5_vehicle_makes_speed_offences(self, output_path, output_format):
        """
        Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences,
        has licensed Drivers, used top 10 used vehicle colours and has car licensed with the Top 25 states
        with highest number of offences (to be deduced from the data)
        :param output_format: Write file format
        :param output_path: output file path
        :return List of Vehicle brands
        """
        top25_offences_state = [row[0] for row in self.df_units.filter(col("VEH_LIC_STATE_ID").cast("int").isNull()). \
                                groupby("VEH_LIC_STATE_ID").count().orderBy(col("count").desc()).limit(25).collect()]

        top10_vehicle_colors = [row[0] for row in self.df_units.filter(col('VEH_COLOR_ID') != "NA"). \
                                groupby("VEH_COLOR_ID").count().orderBy(col("count").desc()).limit(10).collect()]

        df = self.df_charges.join(self.df_primperson, on=['CRASH_ID'], how='inner'). \
            join(self.df_units, on=['CRASH_ID'], how='inner'). \
            filter(self.df_charges.CHARGE.contains("SPEED")). \
            filter(self.df_primperson.DRVR_LIC_TYPE_ID.isin(["DRIVER LICENSE", "COMMERCIAL DRIVER LIC."])). \
            filter(self.df_units.VEH_COLOR_ID.isin(top10_vehicle_colors)). \
            filter(self.df_units.VEH_LIC_STATE_ID.isin(top25_offences_state)). \
            groupby("VEH_MAKE_ID").count(). \
            orderBy(col("count").desc()).limit(5).drop("count")

        utils.write_output(df, output_path, output_format)

        return [row[0] for row in df.collect()]

if __name__ == '__main__':
    # Initialize sparks session
    spark = SparkSession \
            .builder \
            .appName("CarCrash_Spark_Application") \
            .getOrCreate()

    config_file_path = "config.yaml"
    spark.sparkContext.setLogLevel("ERROR")

    CC = CarCrash_Spark_Application(config_file_path)
    output_file_paths = utils.read_yaml(config_file_path).get("OUTPUT_PATH")
    file_format = utils.read_yaml(config_file_path).get("FILE_FORMAT")

    # 1: Find the number of crashes (accidents) in which number of males killed are greater than 2?
    print("Output 1:", CC.count_crash_male_more_than_2(output_file_paths.get(1), file_format.get("Output")))

    # 2: How many two wheelers are booked for crashes?
    print("Output 2:", CC.count_2_wheeler_accidents(output_file_paths.get(2), file_format.get("Output")))

    # 3: Determine the Top 5 Vehicle Makes of the cars present in the crashes in which driver died
    # and Airbags did not deploy.
    print("Output 3:", CC.get_top_vehmake_driver_died_airbags_notdeployed(output_file_paths.get(3),file_format.get("Output")))

    # 4: Determine number of Vehicles with driver having valid licences involved in hit and run?
    print("Output 4:", CC.get_no_vehicles_with_license_hit_run(output_file_paths.get(4), file_format.get("Output")))

    # 5: Which state has highest number of accidents in which females are not involved?
    print("Output 5:"), CC.get_state_highest_accident_no_female(output_file_paths.get(5), file_format.get("Output"))

    # 6: Which are the Top 3rd to 5th VEH_MAKE_IDs that contribute to a largest number of injuries including death
    print("Output 6:", CC.get_top_vehicle_make_max_injuries(output_file_paths.get(6),
                                                                                file_format.get("Output")))

    # 7: For all the body styles involved in crashes, mention the top ethnic user group of each unique body style
    print("Output 7:", CC.get_top_ethnic_ug_each_body_style(output_file_paths.get(7), file_format.get("Output")))

    # 8: Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols as
    # the contributing factor to a crash (Use Driver Zip Code)
    print("Output 8:", CC.get_top_5_zip_codes_crash_with_alcohol(output_file_paths.get(8), file_format.get("Output")))

    # 9: Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~)
    # is above 4 and car avails Insurance
    print("Output 9:", CC.get_crash_ids_with_no_damage(output_file_paths.get(9), file_format.get("Output")))

    # 10: Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences,
    # has licensed Drivers, used top 10 used vehicle colours and has car licensed with the Top 25 states with
    # highest number of offences (to be deduced from the data)
    print("Output 10:", CC.get_top_5_vehicle_makes_speed_offences(output_file_paths.get(10), file_format.get("Output")))

    spark.stop()
