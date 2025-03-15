
from io import StringIO
import pandas as pd
import requests


def load_visogender(_dataset_name: str):

    # The following function was obtained from
    # https://github.com/oxai/visogender/blob/main/src/template_generator_utils.py#L39
    def load_metadata_to_dict(filepath: str, context: str):
        """
        Opens a file, creates a dictionary of dictionaries with IDX as key, 
        and metadata as values.

        Args:
            filepath: filepath to saved .tsv
            context: the type of image scenario - either occupation/participant (OP) or occupation/object(OO

        Returns:
            Tuple with two dictionary (for each context) of dictionaries with metadata for all images IDX
        """
        op_idx_metadata_dict, oo_idx_metadata_dict = {}, {}

        with open(filepath, "r") as file:
            for enum, line in enumerate(file):
                try:
                    if enum >= 1:
                        values = line.strip().split("\t")
            
                        idx = values[0]
                        sector = values[1]
                        specialisation = values[2]
                        occ = values[3]
                        url = values[5]
                        licence = bool(values[6])
                        occ_gender = values[7]
                        error_code = values[-2]
                        annotator = values[-1]

                        if context == "OP":
                            par = values[4]
                            par_gender = values[8]
                            op_idx_metadata_dict[idx] = {
                                "sector": sector, 
                                "specialisation": specialisation, 
                                "occ" : occ, 
                                "par" : par, 
                                "url" : url, 
                                "licence" : licence,
                                "occ_gender" : occ_gender, 
                                "par_gender" : par_gender, 
                                "annotator": annotator
                            }
                        else:
                            obj = values[4]
                            oo_idx_metadata_dict[idx] = {
                                "sector": sector, 
                                "specialisation": specialisation, 
                                "occ" : occ, 
                                "obj" : obj, 
                                "url" : url, 
                                "licence" : licence, 
                                "occ_gender" : occ_gender,
                                "annotator": annotator
                            }
                except IndexError:
                    continue
            
        return op_idx_metadata_dict, oo_idx_metadata_dict
    
    urls = [
        "https://github.com/oxai/visogender/blob/main/data/visogender_data/OO/OO_Visogender_02102023.tsv",
        "https://github.com/oxai/visogender/blob/main/data/visogender_data/OP/OP_Visogender_02102023.tsv",
        "https://github.com/oxai/visogender/blob/main/data/visogender_data/OP/OP_Visogender_11012024.tsv",
    ]

    for idx, url in enumerate(urls):
        response = requests.get(url)
        if response.ok:
            data = response.text
        # with open(f"visogender_{idx}.csv", "w") as current_segment:
        #     current_segment.write(data)
        #     current_segment.close()
        
        # current_df = pd.read_csv(StringIO(data))

    return load_metadata_to_dict()
