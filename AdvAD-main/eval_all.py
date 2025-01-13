from eval_asr_from_npz import eval_asr_from_npz
from eval_quality_from_npz import eval_quality_from_npz
from eval_fid_from_npz import eval_fid_from_npz
def eval_all(file_name=None, re_logger=True, quant=False, eval_model=None):
    eval_quality_from_npz(file_name=file_name, re_logger=re_logger, quant=quant)
    eval_asr_from_npz(file_name=file_name, re_logger=re_logger, quant=quant, eval_model=eval_model)
    eval_fid_from_npz(file_name=file_name, re_logger=re_logger, quant=quant)

if __name__ == "__main__":
    dir_list = [
        # "adv_examples_Res50_AdvAD"
        # "attack20250110_051505"

    ]
    for dir in dir_list:
        print("*********************** Eval All {} ***********************".format(dir))

        '''Eval normal quanted (8-bit image) results for AdvAD (quant=True)'''
        eval_all(file_name=dir, re_logger=True, quant=True, eval_model="inception_v3")

        '''Eval raw floating-point data w/o quant for AdvAD-X (quant=False)'''
        # eval_all(file_name=dir, re_logger=True, quant=False, model_name="resnet50")

        print("************************ Done ************************")

