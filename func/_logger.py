import os
import logging


def check_log_folder(log_folder):
    try:
        os.mkdir(log_folder)
    except FileExistsError:
        pass


def get_logger(logPath, filename):
    check_log_folder(logPath)

    logger = logging.getLogger(__name__)
    logfile = f"./{logPath}/{filename}"
    # try:
    #     os.remove(logfile)
    # except:
    #     logger.warning(f'{str(datetime.datetime.now().astimezone(datetime.timezone(datetime.timedelta(hours=8))))} : cannot remove {logfile}!')

    logging.basicConfig(
        filename=logfile,
        level=logging.DEBUG,
        encoding="utf-8",
        format="%(asctime)s (%(levelname)s) : %(message)s",
    )
    return logger
