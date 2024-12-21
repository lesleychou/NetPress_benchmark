import logging
import re
from mininet.log import setLogLevel, lg

class MininetLogger:
    """
    A utility class to manage Mininet logging to a file and retrieve log content.
    """
    def __init__(self):
        self.log_file = 'mininet.log'
        self.log_level = logging.INFO
        self.formatter = '%(message)s'

    def setup_logger(self):
        self.clear_handlers()
        with open(self.log_file, 'w'):
            pass

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(self.formatter)
        file_handler.setFormatter(file_formatter)

        lg.addHandler(file_handler)
        lg.setLevel(self.log_level)
        setLogLevel('info')  # Set Mininet log level


    def clear_handlers(self):
        """Removes all handlers from the Mininet logger."""
        while lg.handlers:
            lg.removeHandler(lg.handlers[0])

    def get_log_content(self):
        """Reads and returns the content of the log file."""
        with open(self.log_file, 'r') as f:
            return f.read()

    def read_log_content(self, log_content, iter):
        """
        Analyzes the log content to find packet loss percentage.

        Parameters:
            log_content (str): The content of the log file.
            iter (int): The current iteration.

        Returns:
            bool: True if success (0% packet loss), False otherwise.
        """
        match = re.search(r'(\d+)%', log_content)  
        if match:
            number = int(match.group(1))  
            if number == 0:
                print(f"Success in {iter} iterations")  
                return True
        else:
            print("No '%' found in log content.")
        return False
