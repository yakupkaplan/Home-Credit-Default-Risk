# HOME CREDIT DEFAULT RISK RUNNER FUNCTION

import gc
import time
from contextlib import contextmanager
import warnings

from scripts.helper_functions import get_namespace
from scripts.preprocessing import application_train_test, bureau_and_balance, previous_applications, pos_cash, \
    installments_payments, credit_card_balance
from scripts.train import kfold_lightgbm

warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def main(debug=False):
    num_rows = 10000 if debug else None

    with timer("Pre-Processing"):

        # application_train_test
        df = application_train_test(num_rows)

        # bureau & bureau_balance
        bureau = bureau_and_balance(num_rows)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau

        # previous_applications
        prev = previous_applications(num_rows)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev

        # posh_cash
        pos = pos_cash(num_rows)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos

        # installments_payments
        ins = installments_payments(num_rows)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins

        # credit_card_balance
        cc = credit_card_balance(num_rows)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc

        # saving final dataframes
        train_df = df[df['TARGET'].notnull()]
        test_df = df[df['TARGET'].isnull()]
        train_df.to_pickle("data/final_train_df.pkl")
        test_df.to_pickle("data/final_test_df.pkl")
        del train_df, test_df
        gc.collect()

    with timer("Run LightGBM"):
        feat_importance = kfold_lightgbm(df, debug=debug)


if __name__ == "__main__":
    namespace = get_namespace()
    with timer("Full model run"):
        main(debug=namespace.debug)

