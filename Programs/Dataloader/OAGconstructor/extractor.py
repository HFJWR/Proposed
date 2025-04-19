import json
from tqdm import tqdm
import time
import pandas as pd

def extract_references(jsondir, start_file, end_file):
    """
    中身を確認する関数
    jsonfile: 確認したいファイルまでの相対パス
    outputfile: 出力パス
    https://www.aminer.cn/oag-2-1
    """
    json_list = []
    for i in range(start_file, end_file):
        json_list.append(str(jsondir + "mag_papers_" + str(i) + ".txt"))
    
    label = ["paper_id", "title", "year", "citation", "fos", "authors_id", "references"]
    data_list = []
    for jsonfile in json_list:    
        with open(jsonfile, 'rt') as f:
            for l, line in enumerate(tqdm(f)):       
                data = json.loads(line)
                if ("references" in data) and (data["n_citation"]>=20) and(len(data["references"])>=10):    # references参考文献が存在するか
                    if "fos" in data:   # 研究分野が存在するか
                        paper_id = data["id"]
                        title = data["title"]
                        year = data["year"]
                        n_citation = data["n_citation"]

                        authors = data["authors"]
                        authors_id = []
                        for a in authors:
                            authors_id.append(a["id"])
                            
                        references = data["references"]
                        
                        fos = data["fos"]
                        w =0
                        for f in fos:
                            if f["w"]>w:
                                w = f["w"]
                                f_name = f["name"]    
                        
                        data_list.append([paper_id, title, year, n_citation, str(f_name), str(authors_id), str(references)])
                        
    papers = pd.DataFrame(data_list, columns = label)

    return papers

