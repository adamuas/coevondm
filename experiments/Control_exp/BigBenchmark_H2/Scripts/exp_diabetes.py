import sys;
sys.path.insert(0, "../../../optim");
sys.path.insert(0, "../../../datasets");
sys.path.insert(0, "../../../core");
sys.path.insert(0, "../../../visualisation");
sys.path.insert(0, "../../../tools");

import ndmCoevoOptim;
from benchmark import proben1_bechmark as proben1;
from benchmark import lab_bencmark;
import kfold;
import traceback;
import notifcation as notify;

def main(RUNS = 10, numH = 2):
    """
    Daibetes
    """

    # try:
    print ">>STARTING...";
    proben = proben1();
    D = proben.diabetes();

    DCrossVal = kfold.kfold(D = D['train'], numFolds = RUNS);
    netConfig = {'numI': D['train']['INFO']['num_inputs'],
                 'numO': D['train']['INFO']['num_outputs'],
                 'numH': numH
    };
    for ri in xrange(RUNS):
        print ">>>> RUN {0} of {1}".format(ri, RUNS);
        print "ON :", D['name'];
        coevo = ndmCoevoOptim.ndmCoevoOptim(dataset_name = D['NAME'],
                                            train_set = DCrossVal[ri][0],
                                            valid_set = DCrossVal[ri][1],
                                            test_set = D['test'],
                                            netConfig = netConfig);
        coevo.init_populations();
        coevo.coevolve();

    #     #send notification
    #     notify.noticeEMail(D['name']+' DONE');
    # except:
    #     """ """
    #     print "ERROR";
    #     notify.noticeEMail('Diabetes '+' ERROR');


if __name__ == '__main__':
    main();

