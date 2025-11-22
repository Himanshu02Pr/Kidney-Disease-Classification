import os
import subprocess
from Classifier import logger
from Classifier.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the kaggle source
        '''     
        data_source = self.config.source
        download_dir = self.config.download_path

        os.makedirs(download_dir, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = download_dir 

        logger.info(f"Downloading data from {data_source} into file {download_dir}")        

        
        try:
            import kagglehub
        except ImportError:
            print("Installing kaggle...")
            subprocess.run(['pip','install','-q', 'kaggle'], check=True)
            import kagglehub

        try:
            print("Downloading via Kagglehub...")
            kagglehub.dataset_download(data_source)
            print(f"Downloaded Successfully at: {download_dir}")
            
        except Exception as e:
            print("Downloading failed, trying via Kaggle API")
            print(e)
            os.makedirs("~/.kaggle/", exists_ok=True)
            kgl_dir = os.path.expanduser("~/.kaggle/")
            json_path = os.path.join(kgl_dir, "kaggle.json")

            try:
                from tkinter import filedialog
                upload = filedialog.askopenfile(title="Select kaggle.json file",
                                                filetypes=[("JSON files","*.json")])
                if "kaggle.json" in upload:
                    os.system(f"cp kaggle.json {kgl_dir}")
                    os.chmod(json_path, 0o600)
                    print("kaggle.json successfully uploaded!!")
                    print(f"Downloading {data_source}...")
                    command = ["kaggle","datasets","download","-d",data_source,
                                "-p",download_dir]
                    subprocess.run(command, check=True)
                    print(f"Dataset downloaded and extracted in: {download_dir}")

                else:
                    raise FileNotFoundError("kaggle.json file not uploaded")
            except Exception as e:
                raise RuntimeError("Kaggle API Authentication failed.") from e
           
        logger.info(f"Downloaded data from {data_source} into file {download_dir}")