from data.pubtator.pubtator_processor import MultiPubtatorProcessor

default_file_name_list = ['cellline2pubtatorcentral', 'chemical2pubtatorcentral', 'disease2pubtatorcentral', 'gene2pubtatorcentral', 'mutation2pubtatorcentral', 'species2pubtatorcentral']

class PubTatorDataset:
    def __init__(self, file_name_list = default_file_name_list) -> None:
        self.file_name_list = file_name_list
        self.multi_pub_processor = MultiPubtatorProcessor(list_of_file_name=self.file_name_list)
        self.file_name = self.multi_pub_processor.merged_file_name
        self.mention2PMIDs_pickle_file_name = f"{self.file_name}_mention2PMIDs.pickle"
        
    def load_mention2PMIDs(self):
        return self.multi_pub_processor.load_mention2PMIDs_from_saved_file()
    
    