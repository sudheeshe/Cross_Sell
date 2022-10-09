from datetime import datetime
import os

class AppLogger:

    def log(self,file_name, log_msg):
        self.now = datetime.now()
        self.current_time = self.now.strftime('%H:%M:%S')
        self.date = self.now.date()

        file_name.write(str(self.date) + "/" + str(self.current_time) + "\t\t" + log_msg + "\n")

