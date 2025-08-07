import SOC_lib
import time
from alcor_config_lib import Alcor_config
from config2 import arcadia_md_config
    
class alcor_interface:
    
    def __init__(self, alcor_number):
        self.configuration = Alcor_config(alcor_id=0)
        self.alcor_number = alcor_number
        self.memory_controller = SOC_lib.axi_mem_interface(logging=False, log_path="log/memory_interaction_log.txt")
        self.GPIO_controller = SOC_lib.GPIO_controller(self.memory_controller)
        self.DUT_interface_controller = SOC_lib.DUT_interface_controller(self.memory_controller)
        self.serial_controller = SOC_lib.serial_controller(self.memory_controller, 0)
        ## Serial controller settings
        self.serial_controller.serial_mode = 0b010 # Serial mode SPI
        self.serial_controller.nbytes = 3  # 24 bits configuration registers
        self.serial_controller.spi_mode = 0b01#clock polarity, clock phase
        self.serial_controller.clk_divivder = 0x8 # 8*4 = 32, 300 MHz/32 MHz= 9.3 MHz 

        self.verify_configuration = True


    def hard_reset_alcor(self):
        self.DUT_interface_controller.write_control_word(0,1)
        time.sleep(0.001) # millisecond sleep to ensure reset
        self.DUT_interface_controller.write_control_word(0,0b0)
        
    def soft_reset_alcor(self):
        self.DUT_interface_controller.write_control_word(0,2)
        time.sleep(0.001) # millisecond sleep to ensure reset
        self.DUT_interface_controller.write_control_word(0,0b0)


    def write_serial_data(self, alcor_control_word):
        self.serial_controller.data_word_0 = alcor_control_word ## Bit flip
        # print (f"Ser ctr word: {self.serial_controller.control_word:32b}")
        self.serial_controller.start_serial_com()
        # self.serial_controller.read_ser_status()
        # self.serial_controller.read_ser_data_word()

    alcor_lib.py
    def read_last_serial_word(self):
        self.serial_controller.read_ser_data_word()
        # print (f"load={self.serial_controller.return_word_0&0xffff}")
        return (self.serial_controller.return_word_0) # To be tailored
    
    def assemble_control_word(self, read_not_write, SPI_command, payload):
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
        # control_word = ~control_word & 0xFFFFFF
        return control_word

    
    def get_SPI_status(self):
        self.write_serial_data(self.assemble_control_word(1, "SPI_status", 0))
        return self.read_last_serial_word()
    
    def write_SPI_status(self):
        self.write_serial_data(self.assemble_control_word(0, "SPI_status", 0xA1A1))
        return self.read_last_serial_word()


    def get_EoC_status(self):
        self.write_serial_data(self.assemble_control_word(1, "EoC_status", 0))
        return self.read_last_serial_word()
    
    def reset_EoC_status(self):
        self.write_serial_data(self.assemble_control_word(0, "EoC_status", 0))
    
    def write_data_in_addr(self, addr, data):
        self.write_serial_data(self.assemble_control_word(0, "SPI_pointer_reg", addr))
        self.write_serial_data(self.assemble_control_word(0, "SPI_data_reg", data))
        if self.verify_configuration:
            self.write_serial_data(self.assemble_control_word(0, "SPI_pointer_reg", addr))
            self.write_serial_data(self.assemble_control_word(1, "SPI_data_reg", 0x0))
            try:
                assert data==(self.read_last_serial_word() & 0xffff)
            except AssertionError as E:
                print (f"Wrote {data}, read {self.read_last_serial_word()  & 0xffff} in addr {addr}")

    def configure_pixel(self, pixel_id):
        pcr_conf_tuples = self.configuration.prepare_pixel_configuration(pixel_id)
        for addr,data in pcr_conf_tuples:
            self.write_data_in_addr(addr, data)

    def configure_sector(self, sector_id):
        bcr_conf_tuples = self.configuration.prepare_sector_configuration(sector_id)
        for addr,data in bcr_conf_tuples:
            self.write_data_in_addr(addr, data)

    def configure_column(self, sector_id):
        column_conf_tubple = self.configuration.prepare_column_configuration(sector_id)
        self.write_data_in_addr(column_conf_tubple[0], column_conf_tubple[1])
    
    def configure_alcor(self):
        for sector in range(4):
            self.configure_sector(sector)
        for column in range(8):
            self.configure_column(column)
        for pixel in range (32):
            self.configure_pixel(pixel)

    def load_data_delays(self):
        delay_conf_tuples = self.configuration.prepare_lanes_delay()
        # print (f" Address: {delay_conf_tuples[0][0]:#x}, Data: {delay_conf_tuples[0][1]:#x}")
        for addr, data in delay_conf_tuples:
            self.DUT_interface_controller.write_control_word(addr, data)

    def enable_data_com(self):
        data_activation_touple = self.configuration.prepare_lane_activation(True)
        self.DUT_interface_controller.write_control_word(data_activation_touple[0], data_activation_touple[1])


    def configure_arcadia_biases(self):
        arcadia_conf = arcadia_md_config()
        arcadia_conf.write_bias_config(sec_id=None, alcor_if=self)

if __name__ == "__main__":
    alcor_0 = alcor_interface(0)
    alcor_0.configuration.load_alcor_config_from_ini("alcor_config_defaults.ini")

    alcor_0.hard_reset_alcor()
    alcor_0.soft_reset_alcor()
    alcor_0.configure_alcor()
    alcor_0.configure_arcadia_biases()
    alcor_0.load_data_delays()
    alcor_0.enable_data_com()
input("Press Enter to exit...")


