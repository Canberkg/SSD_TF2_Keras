import os
import json
import xmltodict
from Config import cfg_300

xml_filepath    =   cfg_300["XML_FILEPATH"]
json_filepath   =   cfg_300["JSON_FILEPATH"]

xml_list=os.listdir(xml_filepath)
for i in range(len(xml_list)):
    with open(os.path.join(xml_filepath,xml_list[i])) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
        xml_file.close()


        json_data = json.dumps(data_dict)

        json_filename="{}.json".format(xml_list[i].split('.')[0])
        with open(os.path.join(json_filepath,json_filename), "w") as json_file:
            json_file.write(json_data)
            json_file.close()