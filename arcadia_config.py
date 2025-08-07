import SOC_lib  
import time
import json
from datetime import datetime
import os
import pickle

from config import bias_config, global_config, FPGA_config

# Comment
class arcadia_interface:

    """
    Full ARCADIA configuration, global and biases, mask, injection
    """

    def __init__(self, arcadia_number):
        # Initialize Arcadia configuration and interfaces
        self.arcadia_number = arcadia_number
        #self.configuration = Arcadia_config(arcadia_id=0)
        # Global configuration (unchanged for all sectors)
        self.arcadia_global_conf = global_config()

        # Per-sector bias configuration
        self.arcadia_bias_conf = []
        for i in range(16):
            self.arcadia_bias_conf.append(bias_config(i))

        # FPGA configuration
        self.FPGA_conf = FPGA_config()

        # Instantiate interface objects
        self.memory_controller = SOC_lib.axi_mem_interface(logging=False, log_path="log/memory_interaction_log.txt")
        self.GPIO_controller = SOC_lib.GPIO_controller(self.memory_controller)
        self.DUT_interface_controller = SOC_lib.DUT_interface_controller(self.memory_controller)
        self.serial_controller = SOC_lib.serial_controller(self.memory_controller, 0)

        # Set up SPI communication parameters
        self.serial_controller.serial_mode = 0b010  # SPI mode
        self.serial_controller.nbytes = 3           # 24-bit registers
        self.serial_controller.spi_mode = 0b11      # Clock polarity and phase
        self.serial_controller.clk_divivder = 0x8   # Set clock divider for SPI clock

        self.verify_configuration = True  # Enable verification after writing
        self.masked_pixels=[] # List of masked pixels



    def hard_reset_arcadia(self):
        # Perform hard reset using control word
        self.DUT_interface_controller.write_control_word(0, 1)
        time.sleep(0.001) # millisecond sleep to ensure reset
        self.DUT_interface_controller.write_control_word(0, 0)

    def soft_reset_arcadia(self):
        # Perform soft reset using control word
        self.DUT_interface_controller.write_control_word(0, 2)
        time.sleep(0.001) # millisecond sleep to ensure reset
        self.DUT_interface_controller.write_control_word(0, 0)

    def write_serial_data(self, arcadia_control_word):
        # Send control word to serial interface
        self.serial_controller.data_word_0 = arcadia_control_word
        self.serial_controller.start_serial_com()

    def read_last_serial_word(self):
        # Read the last word received from SPI bus
        self.serial_controller.read_ser_data_word()
        return self.serial_controller.return_word_0

    def assemble_control_word(self, read_not_write, SPI_command, payload):
        # Build 24-bit SPI control word
        SPI_command_map = {
            "ICR1_reg":        0b100,
            "SPI_status":      0b010,
            "SPI_pointer_reg": 0b000,
            "SPI_data_reg":    0b001,
            "ICR0_reg":        0b011
        }
        if SPI_command not in SPI_command_map:
            raise ValueError(f"Invalid SPI command: {SPI_command}")

        SPI_command_code = SPI_command_map[SPI_command]
        control_word = ((read_not_write & 0x1) << 23) | \
                       ((SPI_command_code & 0x7) << 20) | \
                       (0b0000 << 16) | \
                       (payload & 0xFFFF)

        if SPI_command == "SPI_pointer_reg":
            control_word = control_word + (1<<13)
        inverted = ~control_word & 0xFFFFFF
        return inverted

    def get_SPI_status(self):
        # Read SPI status register
        self.write_serial_data(self.assemble_control_word(1, "SPI_status", 0))
        return self.read_last_serial_word()

    def write_SPI_status(self):
        # Write predefined value to SPI status register
        self.write_serial_data(self.assemble_control_word(0, "SPI_status", 0xA1A1))
        return self.read_last_serial_word()
    
    def write_ICR0(self, icr0_payload):
        # Write predefined value to ICR0 status register
        self.write_serial_data(self.assemble_control_word(0, "ICR0_reg", icr0_payload))
        return self.read_last_serial_word()

    
    def enable_config(self):
        self.write_ICR0(1 << 8) #it enables the configuration (ICR=[8]=1)
        print("!!!CONFIGURED!!!")

    def write_data_in_addr(self, addr, data):
        # Write data to specific address over SPI
        self.write_serial_data(self.assemble_control_word(0, "SPI_pointer_reg", addr))
        #print('ADDR: ', (self.assemble_control_word(0, "SPI_pointer_reg", addr)))
        self.write_serial_data(self.assemble_control_word(0, "SPI_data_reg", data))
        #print('DATA: ', (self.assemble_control_word(0, "SPI_data_reg", data)))
       
        if self.verify_configuration:
            # punta l'indirizzo da leggere
            self.write_serial_data(self.assemble_control_word(0, "SPI_pointer_reg", addr))
            
            #invia comando di lettura
            self.write_serial_data(self.assemble_control_word(1, "SPI_data_reg", 0x0))
    
            # leggi la risposta aggiornata dallo SPI
            self.serial_controller.read_ser_data_word()
            read_val = ~self.serial_controller.return_word_0 & 0xFFFF  
            #print('!!!CHECK!!!')
            #print('READ VAL: ', self.serial_controller.return_word_0)
            #print('DATA :', data)

            if read_val != data:
                print(f"MISMATCH @0x{addr:04X}: wrote 0x{data:04X}, read 0x{read_val:04X}")
            else:
                print(f"Verified @0x{addr:04X}: 0x{data:04X}")



################################################################################################################
########################      INITIALIZE AND RESET      ########################################################
################################################################################################################


    def fixed_reset_pulse(self):
        self.write_ICR0(1)




    def reset_pulse(self,t=None):
        """
        t = reset pulse duration (seconds)
        """
        if (t==None):
            self.write_ICR0(1)
        else: 
            self.write_ICR0(1<<1)
            time.sleep(t)
            self.write_ICR0(1<<2)
        
        

    def enable_clock_gating(self):
        self.update_global_registers({"SECTION_CLOCK_GATING": 0b1})



    #Startup procedure for the arcadia
    def startup_procedure(self):
        self.update_global_registers({"SECTION_CLOCK_MASK": 0x0000})        # Enable the clock
        self.reset_pulse(t=0.0005)                                          # 50 Clock Cycles(?)
        self.update_global_registers({"DIGITAL_INJECTION": 0xFFFF})         # Enable injection on all the Sections
        self.full_arcadia_config_def()                                      # Enable force configuration on all Sections
        self.update_global_registers({"SECTION_CLOCK_MASK": 0xFFFF})        # Disable Clock to all Sections
        self.update_global_registers({"DIGITAL_INJECTION": 0xFFFF})         # Test Pulse on all Sections
        self.update_global_registers({"SECTION_CLOCK_MASK": 0x0000})        # Enable Clock to all Sections
       
        
        
        
        

    def load_data_delays(self):
        # Load delay values for data lanes
        delay_conf_tuples = self.configuration.prepare_lanes_delay()
        for addr, data in delay_conf_tuples:
            self.DUT_interface_controller.write_control_word(addr, data)

    def enable_data_com(self):
        # Activate data communication lanes
        data_activation_tuple = self.configuration.prepare_lane_activation(True)
        self.DUT_interface_controller.write_control_word(data_activation_tuple[0], data_activation_tuple[1])




    
    def write_global_bias_config(self):
        address_data_map = {}

        for param_name in self.arcadia_global_conf.bias_config_dict_map:
            param_info = self.arcadia_global_conf.bias_config_dict_map[param_name]
            address = param_info["word_address"]
            mask = param_info["mask"]
            offset = param_info["offset"]
            param_value = self.arcadia_global_conf.bias_config[param_name]

            if address not in address_data_map:
                address_data_map[address] = 0

            address_data_map[address] |= (param_value << offset) & mask

        for address, data in address_data_map.items():
            self.write_data_in_addr(address, data)

    # Writes the biases in a section
    def write_bias_config_sect(self, sec_id):
        address_data_map = {}

        for param_name in self.arcadia_bias_conf[sec_id].bias_config_dict_map:
            param_info = self.arcadia_bias_conf[sec_id].bias_config_dict_map[param_name]
            address = param_info["word_address"]
            mask = param_info["mask"]
            offset = param_info["offset"]
            param_value = self.arcadia_bias_conf[sec_id].bias_config[param_name]

            if address not in address_data_map:
                address_data_map[address] = 0
            address_data_map[address] |= (param_value & mask) << offset

        for address, data in address_data_map.items():
            self.write_data_in_addr(address, data)


    # Writes the biases in all the sections
    def write_all_bias_config(self, sec_id=None):
        if sec_id is None:
            for i in range(16):
                self.write_bias_config_sect(i)
        else:
            self.write_bias_config_sect(sec_id)





################################################################################################################
########################   FULL ARCADIA CONFIGURATION   ########################################################
################################################################################################################

    
    def configure_arcadia_biases(self):
        self.write_global_bias_config()
        print("GLOBAL CONFIGURATION MASTER DONE")
        self.write_all_bias_config()


    # configures both master and slave
    def full_arcadia_config_def(self):
        self.configure_arcadia_biases()
        self.enable_config() #it enables the configuration
        self.update_global_registers({"HELPER_SECCFG_PIXELSELECT": 0x0F}) # Configures the slave
        
        
        
        
        


################################################################################################################
########################   GLOBAL   ############################################################################
################################################################################################################

    def update_global_registers(self, dict_to_update):
        """
        Updates the global configuration registers using provided dictionary.
        """

        # Validate and apply the new parameter values
        for key, val in dict_to_update.items():
            if key not in self.arcadia_global_conf.bias_config:
                print(f"Key '{key}' not found in global configuration. Skipping.")
            else:
                self.arcadia_global_conf.bias_config[key] = val
                
        
        self.write_global_bias_config() # We configure all the global to avoid overwriting 
        self.enable_config()
        print(">>> Global registers updated.")



################################################################################################################
########################   BIAS   ##############################################################################
################################################################################################################


    def update_biases(self, dict_to_update, sec_id=None):
        """
        Aggiorna i registri di bias per master e slave.
        - dict_to_update: dizionario con i parametri da aggiornare (es. {"VCASN": 10})
        - sec_id: se specificato, aggiorna solo il settore indicato; altrimenti tutti
        """
        
        # Validate and apply the new parameter values
        
        
        if sec_id == None:
            for sec in range(16):
                for key, val in dict_to_update.items():
                    if key not in self.arcadia_bias_conf[sec].bias_config:
                        print(f"Key '{key}' not found in global configuration. Skipping.")
                    else:
                        self.arcadia_bias_conf[sec].bias_config[key] = val
            self.full_arcadia_config_def()
        else:
            for key, val in dict_to_update.items():
                if key not in self.arcadia_bias_conf[sec_id].bias_config:
                    print(f"Key '{key}' not found in global configuration. Skipping.")
                else:
                    self.arcadia_bias_conf[sec_id].bias_config[key] = val
            
            self.update_global_registers({"HELPER_SECCFG_PIXELSELECT": 0x1F})
            self.write_bias_config_sect(sec_id)
            self.enable_config() #it enables the configuration
            self.update_global_registers({"HELPER_SECCFG_PIXELSELECT": 0x0F}) # Configures the slave
            self.write_bias_config_sect(sec_id)
            self.enable_config() #it enables the configuration
            

        print(f">>> Bias aggiornati per {'settore ' + str(sec_id) if sec_id is not None else 'tutti i settori'} (master & slave)")




################################################################################################################
########################   MASK   ##############################################################################
################################################################################################################


    def mask_pixel(self, sect, col, prstart, prskip, prstop, pixelsel):

        self.masked_pixels.append({
            "sect": sect,
            "col": col,
            "prstart": prstart,
            "prskip": prskip,
            "prstop": prstop,
            "pixelsel": pixelsel
        })

        self.update_global_registers({"HELPER_SECCFG_SECTIONS": sect})
        self.update_global_registers({"HELPER_SECCFG_COLUMNS": col})
        self.update_global_registers({"HELPER_SECCFG_PRSTART": prstart})
        self.update_global_registers({"HELPER_SECCFG_PRSKIP": prskip})
        self.update_global_registers({"HELPER_SECCFG_CFGDATA": 0b10}) #mask
        self.update_global_registers({"HELPER_SECCFG_PRSTOP": prstop})
        self.update_global_registers({"HELPER_SECCFG_PIXELSELECT": pixelsel})


    def unmask_pixel(self, sect, col, prstart, prskip, prstop, pixelsel):
        # Remove matching pixel from the masked list
        """
        Now not working
        """
        before = len(self.masked_pixels)
        self.masked_pixels = [
            p for p in self.masked_pixels
            if not (
                p["sect"] == sect and
                p["col"] == col and
                p["prstart"] == prstart and
                p["prskip"] == prskip and
                p["prstop"] == prstop and
                p["pixelsel"] == pixelsel
            )
        ]
        after = len(self.masked_pixels)
        print(f">>> Unmasked pixel. Removed {before - after} matching entry(ies).")

        # Apply the actual unmasking
        self.update_global_registers({"HELPER_SECCFG_SECTIONS": sect})
        self.update_global_registers({"HELPER_SECCFG_COLUMNS": col})
        self.update_global_registers({"HELPER_SECCFG_PRSTART": prstart})
        self.update_global_registers({"HELPER_SECCFG_PRSKIP": prskip})
        self.update_global_registers({"HELPER_SECCFG_CFGDATA": 0b00})  # unmask NOT ACTUAL VALUE
        self.update_global_registers({"HELPER_SECCFG_PRSTOP": prstop})
        self.update_global_registers({"HELPER_SECCFG_PIXELSELECT": pixelsel})


    def inject_pixel(self, sect, col, prstart, prskip, prstop, pixelsel):

        self.masked_pixels.append({
            "sect": sect,
            "col": col,
            "prstart": prstart,
            "prskip": prskip,
            "prstop": prstop,
            "pixelsel": pixelsel
        })

        self.update_global_registers({"HELPER_SECCFG_SECTIONS": sect})
        self.update_global_registers({"HELPER_SECCFG_COLUMNS": col})
        self.update_global_registers({"HELPER_SECCFG_PRSTART": prstart})
        self.update_global_registers({"HELPER_SECCFG_PRSKIP": prskip})
        self.update_global_registers({"HELPER_SECCFG_CFGDATA": 0b01}) #injection
        self.update_global_registers({"HELPER_SECCFG_PRSTOP": prstop})
        self.update_global_registers({"HELPER_SECCFG_PIXELSELECT": pixelsel})


################################################################################################################
########################   ENCODING AND DECODING   #############################################################
################################################################################################################


    # Decimal ---> one hot encoder
    def dec_to_one_hot(self, value,no_of_bits):
        
        encoded_value = 0b0
        encoded_value = encoded_value | (1<<value)
        encoded_value = self.reverse_bits(encoded_value,no_of_bits)
        return encoded_value

    # One hot ---> decimal decoder
    def one_hot_to_dec(self, value, no_of_bits):
        value=self.reverse_bits(value, no_of_bits) 
        number=[]
        for i in range(no_of_bits):
            if (value >> i) & 1 == 1:
                number.append(i)
                
        
        return number


    # Reverse the bits of a word. Example: 0b10100 ---> 0b00101
    def reverse_bits(self,n, no_of_bits):
        """
        n: word to reverse
        no_of_bits: number of bits of the word n
        """
        result = 0
        for i in range(no_of_bits):
            result <<= 1
            result |= n & 1
            n >>= 1
        return result



    #Convert the pixel address(sec,col,or,pxl) to the matrix(x,y)
    def address_to_matrix(self,sect,col,pr,pixel):
        """
        sect: one hot starting from MSB
        col: one hot starting from MSB
        pr: starts from the bottom in decimal (0-63) (TO BE FIXED)
        
        pixel: |0|1|
               |3|2|    one hot encoded starting frm MSB. Master on bottom, slave on top
               
        Example: (0b1000000000000000, 0b1000000000000000, 0, 0b10001) ---> [0,0]
        
        """
        if not (0 <= sect <= 0b1111111111111111):
            raise ValueError("sect must be a 16-bit word .")
        if not (0 <= col <= 0b1111111111111111):
            raise ValueError("col must be a 16-bit word.")
        if not (0 <= pr <= 0b1111111):
            raise ValueError("pr must be a 7-bit word.")
        if not (0 <= pixel <= 0b11111):
            raise ValueError("Input must be a 5-bit word.")
        

        ms = (pixel >> 4) & 0b1           # Extract the MSB (bit 4), Master(1) or Slave(0)
        pix_position = pixel & 0b1111     # Extract the last 4 bits (bits 0-3)

        # Convert from one hot to decimal
        sect=self.one_hot_to_dec(sect,16)
        col=self.one_hot_to_dec(col,16)
        pix_position=self.one_hot_to_dec(pix_position,4)
        #print('pix_position: ', pix_position)
        pixelx=[]
        pixely=[]
        
        # Calculate pixel x coordinate
        #print('len pix_position: ', len(pix_position))
        
        x=[]
        y=[]
        
        for i in range(len(pix_position)):
            print('i: ', i)
            if (pix_position[i] == 1) or (pix_position[i] == 2):
                pixelx.append(1)
            if (pix_position[i] == 0) or (pix_position[i] == 3):
                pixelx.append(0)

            #print('pixelx: ',pixelx,' col: ', col,' sect: ', sect)
        
        
        
            x.append(pixelx[i]+2*col[0]+32*sect[0])
        
        # Calculate pixel y coordinate
            if pix_position[i] in (0,1):
                pixely.append(1)
            if pix_position[i] in (2,3):
                pixely.append(0)
            

            y.append(pixely[i]+4*pr+2*(not ms))
        
        return x, y    




    
    # Convert matrix (x, y) back to pixel address (sect, col, pr, pixel)
    def matrix_to_address(self, x, y):
        if not (0 <= x < 511):
            raise ValueError("x must be in the range [0, 511]")
        if not (0 <= y < 511):
            raise ValueError("y must be in the range [0, 511]")

        # Decode components
        sect = x // 32
        col = (x % 32) // 2
        pixelx = x % 2

        pr = y // 4
        pixely = (y % 4) // 2
        ms = y % 2
        prstart = pr
        prstop = pr
        prskip = 0

        # Determine pixel position from pixelx and pixely
        if pixelx == 0 and pixely == 0:
            pix_position = 3
        elif pixelx == 1 and pixely == 0:
            pix_position = 2
        elif pixelx == 1 and pixely == 1:
            pix_position = 1
        elif pixelx == 0 and pixely == 1:
            pix_position = 0
        else:
            raise ValueError("Invalid pixel (x, y) combination")

        # Combine ms and pix_position into 5-bit pixel
        pixel = (ms << 4) | (1 << pix_position)

        # Convert sect, col, pix_position into one-hot
        sect_oh = self.dec_to_one_hot(sect,16)
        col_oh = self.dec_to_one_hot(col,16)


        return sect_oh, col_oh, prstart, prskip, prstop, pixel
        
        
    # mask_pixel(self, sect, col, prstart, prskip, prstop, pixelsel)

################################################################################################################
########################   SAVE   ##############################################################################
################################################################################################################

     #Create a dictionary comprehensive of global_config and 16 bias_config 
    def create_full_dict(self):
        self.arcadia_bias_conf_list=[]
        for i in range(16):
            self.arcadia_bias_conf_list.append(self.arcadia_bias_conf[i].bias_config)
    
    
        
        full_dict= {'global':self.arcadia_global_conf.bias_config,'biases':self.arcadia_bias_conf_list}
        
        return full_dict
    
    
    
    
    #Create a pickle file with our dictionary    
    def dict_to_pickle(self, filename):
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        os.makedirs("pickles", exist_ok=True)
        full_path = os.path.join("pickles", filename)
        # Write to pickle
        full_dict=[]
        full_dict=self.create_full_dict()
        
        
        with open('full_dict.pkl', 'wb') as handle:
            pickle.dump(full_dict, handle)

        print(f"Configuration saved in '{full_path}'")  
    
        



            
    #Read a pickle file and save it as new configuration and configure chip          
    def config_from_pickle(self, filename):

        if not filename.endswith('.pkl'):
            filename += '.pkl'
        full_path = os.path.join("pickles", filename)

        full_dict=[]
        with open('filename.pkl', 'rb') as handle:
            full_dict= pickle.load(handle)
    
        self.arcadia_global_conf.bias_config = full_dict['global']    
         
         
       
        for sec in range(16):               
            self.arcadia_bias_conf[sec].bias_config= full_dict['biases'][sec]
    
        self.full_arcadia_config_def()

        print(f"Configuration configured from '{full_path}'")
    
    



    def save_json(self, filename):
        base_name = os.path.splitext(filename)[0]

        save_dir = "saved_json"
        os.makedirs(save_dir, exist_ok=True)

        # Percorso completo del file .json
        json_path = os.path.join(save_dir, f"{base_name}.json")

        # Costruisci il dizionario da salvare
        data = {
            "global_params": self.arcadia_global_conf.bias_config,
            "bias_param_values": {
                f"sector_{i}": self.arcadia_bias_conf[i].bias_config for i in range(16)
            },
            "pixel_config": getattr(self, "pixel_config", {}),
            "masked_pixels": getattr(self, "masked_pixels", [])
        }

        # Funzione di conversione in formato esadecimale leggibile
        def to_hex_str(obj):
            return {
                k: (f"0x{v:X}" if isinstance(v, int) else v)
                for k, v in obj.items()
            }

        # Applica conversione a tutti i sotto-dizionari
        json_data = {
            k: {sk: to_hex_str(sv) if isinstance(sv, dict) else sv for sk, sv in v.items()} if isinstance(v, dict) else v
            for k, v in data.items()
        }

        # Salvataggio su file
        try:
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=4)
            print(f"[✔] Configurazione JSON salvata in: {json_path}")
        except Exception as e:
            print(f"[ERRORE] Impossibile salvare il file JSON: {e}")


################################################################################################################
########################   ACQUISITION SETTINGS   ##############################################################
################################################################################################################
    def load_data_lanes_delays(self, lane_delay_0=15,lane_delay_1=15,lane_delay_2=15,lane_delay_3=15):
        ## Update stored values
        self.FPGA_conf.FPGA_conf_dict["LANE_DELAY_0"]=lane_delay_0
        self.FPGA_conf.FPGA_conf_dict["LANE_DELAY_1"]=lane_delay_1
        self.FPGA_conf.FPGA_conf_dict["LANE_DELAY_2"]=lane_delay_2
        self.FPGA_conf.FPGA_conf_dict["LANE_DELAY_3"]=lane_delay_3
        address_data_map = {}
        for param_name in ["LANE_DELAY_0","LANE_DELAY_1","LANE_DELAY_2","LANE_DELAY_3"]:
            param_info =self.FPGA_conf.address_map[param_name]
            address = param_info["word_address"]
            mask = param_info["mask"]
            offset = param_info["offset"]
            param_value = self.FPGA_conf.FPGA_conf_dict[param_name]

            if address not in address_data_map:
                address_data_map[address] = 0
            address_data_map[address] |= (param_value & mask) << offset

        for address, data in address_data_map.items():
            self.DUT_interface_controller.write_control_word(address, data)


    def toggle_data_com(self, activate=True):
        self.FPGA_conf.FPGA_conf_dict["ACTIVATE_DATA"]=int(activate)
        address_data_map = {}
        for param_name in ["DDR_MODE","ACTIVATE_DATA"]:
            param_info =self.FPGA_conf.address_map[param_name]
            address = param_info["word_address"]
            mask = param_info["mask"]
            offset = param_info["offset"]
            param_value = self.FPGA_conf.FPGA_conf_dict[param_name]

            if address not in address_data_map:
                address_data_map[address] = 0
            address_data_map[address] |= (param_value & mask) << offset

        for address, data in address_data_map.items():
            self.DUT_interface_controller.write_control_word(address, data)


    def set_axi_packet_length(self, length=64):
        self.FPGA_conf.FPGA_conf_dict["AXI_PACKET_LENGTH"]=length
        address_data_map = {}
        for param_name in ["AXI_PACKET_LENGTH"]:
            param_info =self.FPGA_conf.address_map[param_name]
            address = param_info["word_address"]
            mask = param_info["mask"]
            offset = param_info["offset"]
            param_value = self.FPGA_conf.FPGA_conf_dict[param_name]

            if address not in address_data_map:
                address_data_map[address] = 0
            address_data_map[address] |= (param_value & mask) << offset

        for address, data in address_data_map.items():
            self.DUT_interface_controller.write_control_word(address, data)
    
    def read_8b10b_errors_counter(self):
        """
        Reads the 8b10b errors counter from the FPGA.
        Returns the value of the counter.
        """
        for i in range(0,7):
            self.DUT_interface_controller.write_control_word(5, i)
            time.sleep(0.001)
            errors = self.DUT_interface_controller.read_control_word(0x0)
            print (f"Errors Lane {i*2}:{(errors & 0xFFFF)}, {i*2+1}: {(errors >> 16) & 0xFFFF}")

if __name__ == "__main__":
    # Run full configuration routine when executed as script
    arcadia_0 = arcadia_interface(0)
    #arcadia_0.configuration.load_arcadia_config_from_ini("arcadia_config_defaults.ini")

    arcadia_0.hard_reset_arcadia()
    arcadia_0.soft_reset_arcadia()
    #arcadia_0.test_write(addr=10)
    #arcadia_0.configure_gcr()
    #arcadia_0.configure_arcadia_biases()
    #arcadia_0.enable_config() #it enables the configuration
    #arcadia_0.configure_arcadia_biases_slave()
    #arcadia_0.enable_config() #it enables the configuration
    #arcadia_0.full_arcadia_config_def() #full configuration with default values
    #arcadia_0.update_global_registers({"READOUT_CLK_DIVIDER": 5})
    #arcadia_0.update_biases({"VCASN": 10}, sec_id=3) #To set the biases.
    arcadia_0.startup_procedure()
    arcadia_0.set_axi_packet_length(64)  # Set AXI packet length
    arcadia_0.load_data_lanes_delays()
    arcadia_0.toggle_data_com(activate=True)  # Enable data communication
    arcadia_0.update_global_registers({"SERIALIZER_SYNC":1})
    arcadia_0.read_8b10b_errors_counter()

    #arcadia_0.load_data_delays()
    #arcadia_0.enable_data_com()
