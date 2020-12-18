import logging
import datetime

log_dir = 'logs/'
basename = "mylogfile"

def make_logger(basename=None):
    #1 logger instance를 만든다.
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = log_dir + "_".join([basename, suffix, '.log']) # e.g. 'mylogfile_120508_171442'
    
    logger = logging.getLogger(basename)

    #2 logger의 level을 가장 낮은 수준인 DEBUG로 설정해둔다.
    logger.setLevel(logging.DEBUG)

    #3 formatter 지정
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s\n%(message)s")
    
    #4 handler instance 생성
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=filename)
    
    #5 handler 별로 다른 level 설정
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    #6 handler 출력 format 지정
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    #7 logger에 handler 추가
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger