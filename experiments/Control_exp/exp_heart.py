import sys;
sys.path.insert(0, "../../optim");
sys.path.insert(0, "../../datasets");
sys.path.insert(0, "../../core");
sys.path.insert(0, "../../visualisation");
sys.path.insert(0, "../../tools");

import ndmCoevoOptim;
from benchmark import proben1_bechmark as proben1;
import kfold;
import traceback;
import notifcation as notify;

def main(RUNS = 10, numH = 2):
    """
    Heart
    """

    # try:
    print ">>STARTING...";
    proben = proben1();
    D = proben.heart();

    DCrossVal = kfold.kfold(D = D['train'], numFolds = RUNS);
    netConfig = {'numI': D['train']['INFO']['num_inputs'],
                 'numO': 1,
                 'numH': numH
    };
    for ri in xrange(RUNS):
        print ">>>> RUN {0} of {1}".format(ri, RUNS);
        print "ON :", D['name'];
        coevo = ndmCoevoOptim.ndmCoevoOptim(dataset_name = D['name'],
                                            train_set = DCrossVal[ri][0],
                                            valid_set = DCrossVal[ri][1],
                                            test_set = D['test'],
                                            netConfig = netConfig);
        coevo.init_populations();
        coevo.coevolve();

        #send notification
        #notify.noticeEMail(D['name']+' DONE');

    # except:
    #     """ """
    #     traceback.print_stack();
        #notify.noticeEMail('Heart '+' ERROR');


if __name__ == '__main__':
    main();

