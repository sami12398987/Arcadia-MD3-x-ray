class arcadia_md_config:
    def __init__(self):
        self.global_config = global_config()
        self.bias_config_dict = {}
        for i in range(0, n_sector):
            self.bias_config_dict[i] = bias_config(i)
        
    def write_bias_config_sect(self, sec_id, alcor_if):
        # Group parameters by their word_address
        address_data_map = {}
        for param_name in self.bias_config_dict[sec_id].bias_config_dict_map:
            param_info = self.bias_config_dict[sec_id].bias_config_dict_map[param_name]
            address = param_info["word_address"]
            mask = param_info["mask"]
            offset = param_info["offset"]
            param_value = self.bias_config_dict[sec_id].bias_config[param_name]
            
            # Initialize address entry if not present
            if address not in address_data_map:
                address_data_map[address] = 0
            
            # Apply mask and offset, then accumulate
            address_data_map[address] |= (param_value & mask) << offset
        
        # Write accumulated data to each address
        for address, data in address_data_map.items():
            alcor_if.write_data_in_addr(address, data)

    def write_bias_config(self, sec_id, alcor_if):
        if sec_id is None:
            for i in self.bias_config_dict:
                self.write_bias_config_sect(i, alcor_if)
        else:
            self.write_bias_config_sect(sec_id, alcor_if)

class bias_config:
    def __init__(self, id):
        self.sector_id = id

        self.bias_config = {
            "VCAL_LO":   0,
            "VCAL_HI":   15,
            "VCASD":     4,
            "VCASP":     4,
            "ISF_VINREF": 7,
            "IOTA":      0,
            "VCASN":     33,
            "ICLIP":    1,
            "IBIAS":    2,
            "VREF_LDO":  1,
            "IFB":      2,
            "ISF":      2,
            "BGR_MEAN":  7,
            "BGR_SLOPE": 7,
            "VINREF":    7,
            "ID":       1,
            "LDO_EN":   1,
        }

        self.bias_config_dict_map = {
            "VCAL_LO":    {"word_address": 12 + self.sector_id * 3, "mask": 0x0001, "offset":  0, "default_value":  0},
            "VCAL_HI":    {"word_address": 12 + self.sector_id * 3, "mask": 0x000f, "offset":  1, "default_value": 15},
            "VCASD":      {"word_address": 12 + self.sector_id * 3, "mask": 0x0007, "offset":  5, "default_value":  4},
            "VCASP":      {"word_address": 12 + self.sector_id * 3, "mask": 0x000f, "offset":  8, "default_value":  4},
            "ISF_VINREF": {"word_address": 12 + self.sector_id * 3, "mask": 0x0007, "offset": 12, "default_value":  7},
            "IOTA":       {"word_address": 12 + self.sector_id * 3, "mask": 0x0001, "offset": 15, "default_value":  0},

            "VCASN":      {"word_address": 13 + self.sector_id * 3, "mask": 0x003f, "offset":  0, "default_value": 33},
            "ICLIP":      {"word_address": 13 + self.sector_id * 3, "mask": 0x0003, "offset":  6, "default_value":  1},
            "IBIAS":      {"word_address": 13 + self.sector_id * 3, "mask": 0x0003, "offset":  8, "default_value":  2},
            "VREF_LDO":   {"word_address": 13 + self.sector_id * 3, "mask": 0x0003, "offset": 10, "default_value":  1},
            "IFB":        {"word_address": 13 + self.sector_id * 3, "mask": 0x0003, "offset": 12, "default_value":  2},
            "ISF":        {"word_address": 13 + self.sector_id * 3, "mask": 0x0003, "offset": 14, "default_value":  2},

            "BGR_MEAN":   {"word_address": 14 + self.sector_id * 3, "mask": 0x000f, "offset":  0, "default_value":  7},
            "BGR_SLOPE":  {"word_address": 14 + self.sector_id * 3, "mask": 0x000f, "offset":  4, "default_value":  7},
            "VINREF":     {"word_address": 14 + self.sector_id * 3, "mask": 0x001f, "offset":  8, "default_value":  7},
            "ID":         {"word_address": 14 + self.sector_id * 3, "mask": 0x0003, "offset": 13, "default_value":  1},
            "LDO_EN":     {"word_address": 14 + self.sector_id * 3, "mask": 0x0001, "offset": 15, "default_value":  1},
        }

class global_config:
    pass

print("The file is running correctly.")
input("Press Enter to exit...")

