from pprint import pprint
from datetime import datetime

from ib_insync import IB, util
from ib_insync.contract import Future, Contract


def get_details(**kwargs):
    cont = Future(**kwargs)
    cds = ib.reqContractDetails(cont)
    contracts = [cd.contract for cd in cds]
    details = {key: set(getattr(c, key) for c in contracts)
               for key in Contract.defaults}
    df = util.df(cds)
    print(df.iloc[0])

    try:
        nearest_contract = Future(kwargs['symbol'],
                                  exchange=list(details['exchange'])[0],
                                  lastTradeDateOrContractMonth=find_nearest(details))
        print(nearest_contract)

    except:
        nearest_contract = None

    try:
        df = df[['marketName', 'longName',
                 'validExchanges', 'minTick', 'contractMonth']]
    except:
        df = None

    if nearest_contract:
        hist = util.df(get_history(nearest_contract))

    return details, df, hist


def find_nearest(details):
    dates = [datetime.strptime(contract, '%Y%m%d')
             for contract in details['lastTradeDateOrContractMonth']]
    dates_dict = {max((date - datetime.today()).days, 0):
                  date for date in dates}
    nearest = dates_dict[min(non_zero for non_zero in dates_dict.keys()
                             if non_zero > 0)]
    return nearest.strftime('%Y%m%d')


def get_history(contract):
    return ib.reqHistoricalData(contract,
                                endDateTime='',
                                durationStr='5 D',
                                barSizeSetting='1 DAY',
                                whatToShow='TRADES',
                                useRTH=False,
                                formatDate=1,
                                keepUpToDate=False,
                                )


if __name__ == '__main__':
    import argparse
    # parsing command line arguments
    parser = argparse.ArgumentParser(
        'Prints list of available contracts for given parameters', add_help=True)
    parser.add_argument('symbol', type=str)
    parser.add_argument('-e', '--exchange', nargs='?', type=str)
    parser.add_argument('-c', '--currency', nargs='?', type=str)
    parser.add_argument('-o', '--includeExpired', action='store_false',
                        help='only active contracts (exclude expired)')

    args = vars(parser.parse_args())
    symbol = args['symbol']
    args = {k: v for k, v in args.items() if v is not None}
    # details = get_datails(symbol, **args)
    print(f'Arguments: {args}')

    ib = IB()
    ib.connect('127.0.0.1', 4002, clientId=15)

    details = get_details(**args)
    ib.disconnect()

    pprint(details[0])

    print()
    pprint(details[1])

    print()
    pprint(details[2])
