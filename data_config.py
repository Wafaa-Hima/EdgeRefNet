class DataConfig:
    def __init__(self):
        self.data_name = ""
        self.root_dir = ""
        self.label_transform = ""
    
        self.scheduler = 'onecycle'     
        self.max_lr = 3e-4               
        self.min_lr = 1e-6              
        self.weight_decay = 1e-4         
        self.T_max = 100                
        self.pct_start = 0.3
    
    def get_data_config(self, data_name, custom_root_dir=None):
        self.data_name = data_name
        if custom_root_dir:
            self.root_dir = custom_root_dir
        elif data_name == 'LEVIR-CD-256': 
            self.label_transform = "norm"
            self.root_dir = r'/Data/LEVIR-CD-256' # your dataset path
        elif data_name == 'WHU-CD-256':
            self.label_transform = "norm"
            self.root_dir = r'Data/WHU-CD-256'    # your dataset path
        elif data_name == 'quick_start_LEVIR':
            self.root_dir = './samples_LEVIR'
        elif data_name == 'quick_start_WHU':
            self.root_dir = './samples_WHU'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self

if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR-CD-256')
   