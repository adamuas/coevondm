import sys;
sys.path.insert(0, "../../optim");
sys.path.insert(0, "../../datasets");
sys.path.insert(0, "../../core");
sys.path.insert(0, "../../visualisation");
sys.path.insert(0, "../../tools");

import ndmCoevoOptim;
from benchmark import lab_bencmark;
import kfold;
import notifcation as notify;

def main(RUNS = 10, numH = 2):
    """
    Vertebral Column3C
    """

    # try:
    print ">>STARTING...";
    lb_bench = lab_bencmark();
    D = lb_bench.vertebral_column2C();
    D['name'] = 'Column2C';
    DCrossVal = kfold.kfold(D = D['train'], numFolds = RUNS);
    netConfig = {'numI': 6,
                 'numO': 1,
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

        #send notification
        #notify.noticeEMail(D['name']+' DONE');
    # except:
    #     """ """
    #     # traceback.print_stack();
    #     print "ERROR";
        #notify.noticeEMail('Column2C DONE');


if __name__ == '__main__':
    main();

