from openvqa.datasets.vqa.eval.vqa import VQA
from openvqa.datasets.vqa.eval.vqaEval import VQAEval
import json, pickle
import numpy as np


def eval(__C, dataset, ans_ix_list, pred_list, result_eval_file, ensemble_file, log_file, valid=False):
    result_eval_file = result_eval_file + '.json'

    qid_list = [ques['question_id'] for ques in dataset.ques_list]
    ans_size = dataset.ans_size

    result = [{
        'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],
        # 'answer': dataset.ix_to_ans[ans_ix_list[qix]],
        'question_id': int(qid_list[qix])
    } for qix in range(qid_list.__len__())]

    print('Save the result to file: {}'.format(result_eval_file))
    json.dump(result, open(result_eval_file, 'w'))


    if __C.TEST_SAVE_PRED:
        print('Save the prediction vector to file: {}'.format(ensemble_file))

        pred_list = np.array(pred_list).reshape(-1, ans_size)
        result_pred = [{
            'pred': pred_list[qix],
            'qid': int(qid_list[qix])
        } for qix in range(qid_list.__len__())]

        pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)


    if valid:
        # create vqa object and vqaRes object
        ques_file_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val']]
        ans_file_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val'] + '-anno']

        vqa = VQA(ans_file_path, ques_file_path)
        vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

        # create vqaEval object by taking vqa and vqaRes
        vqaEval = VQAEval(vqa, vqaRes, n=2, target=__C.TARGET)  # n is precision of accuracy (number of places after decimal), default is 2
        # MODIFICATION - __C.TARGET is the trojan target, used to compute ASR, if given.

        # evaluate results
        """
        If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
        By default it uses all the question ids in annotation file
        """
        vqaEval.evaluate()

        # print accuracies and asr
        print("\n")
        print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        # print("Per Question Type Accuracy is the following:")
        # for quesType in vqaEval.accuracy['perQuestionType']:
        #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
        # print("\n")
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")
        if __C.TARGET is not None:
            print("Overall ASR is: %.02f\n" % (vqaEval.asr['overall']))
            print("Per Answer Type ASR is the following:")
            for ansType in vqaEval.asr['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.asr['perAnswerType'][ansType]))
            print("\n")

        # log accuracies and asr
        print('Write to log file: {}'.format(log_file))
        logfile = open(log_file, 'a+')

        # note eval data type in logfile
        if __C.VER != 'clean' and not __C.TROJ_DIS_I:
            logfile.write('Eval Image Data:    TROJ\n')
        else:
            logfile.write('Eval Image Data:    CLEAN\n')
        if __C.VER != 'clean' and not __C.TROJ_DIS_Q:
            logfile.write('Eval Question Data: TROJ\n')
        else:
            logfile.write('Eval Question Data: CLEAN\n')

        logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        for ansType in vqaEval.accuracy['perAnswerType']:
            logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        if __C.TARGET is not None:
            logfile.write("\nOverall ASR is: %.02f\n" % (vqaEval.asr['overall']))
            for ansType in vqaEval.asr['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.asr['perAnswerType'][ansType]))        
        logfile.write("\n\n")
        logfile.close()
