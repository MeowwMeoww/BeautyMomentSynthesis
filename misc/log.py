from tabulate import tabulate as tb
from datetime import date, datetime


def write_log(old_log, new_message, type):
    """
        This function adds messages to the existing log message.
        Parameters
        ----------
        + old_log: str.
            String of old log messages.
        + new_message: str or DataFrame.
        + type: str.
            The message's type. The types are:
                + "string + enter"
                + "dataframe + enter"
        Return
        ----------
            Returns an updated log message.
    """
    
    now = datetime.now()
    date_string = str(date.today()) + "\n"
    time_string = str(str(now.strftime("%H:%M:%S"))) + ": \n"
    
    if type == "string + enter":
        log = old_log + date_string + time_string + new_message + "\n" + "\n"
    elif type == "dataframe + enter": 
        log = old_log + date_string + time_string + tb(new_message, headers='keys', tablefmt="github", numalign="right") + "\n" + "\n"
        
    return log
    

def log_init():
    log = "Filtered beauty moment video generator \n"
    
    return log
    
 
def log_final(log):
    f = open("results/log.txt", "w")
    f.write(log)
    f.close()
    