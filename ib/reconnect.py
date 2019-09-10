   def __init__(self, symbol='CL', contract_month='201910', exchange='NYMEX', database=None,
                 num_decimals=2, min_tick_size=0.01, client_id=None):
        self.ib = None
        self.contract = Future(
            symbol=symbol, lastTradeDateOrContractMonth=contract_month, exchange=exchange, currency='USD')
        if client_id:
            self.client_id = client_id
        else:
            self.client_id = 16
        print('Using client_id: {}'.format(self.client_id))
        self.count = 0

    def connect(self, port=7497):
        self.ib = IB()
        self.ib.connect('127.0.0.1', port, clientId=self.client_id)
        self.ib.errorEvent += self.onErrorEvent

    def onBarUpdate(self, bars, hasNewBar):
        print('.', end='', flush=True)

        self.count += 1
        if self.count > 5:
            self.count = 0

        if hasNewBar:
            print('!', end='', flush=True)

    def make_bars(self):

        bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
            keepUpToDate=True)

        bars.updateEvent += self.onBarUpdate

        return bars

    def onErrorEvent(self, reqId, error_code, error_string, contract):

        # 1.0 Network disconnect
        if error_code in [10182, 1102]:
            logger.info('Cancelling historical data...')
            self.ib.cancelHistoricalData(bars)  # Raise error_code 162

        # 2.0 Historical data
        if error_code in [162]:
            logger.info('Cancelled historical data.  Making new bars...')
            self.run()

    def disconnect(self, msg=None):
        if msg:
            logger.info('Disconnecting for {}...'.format(msg))
        else:
            logger.info('Disconnecting...')
        self.ib.disconnect()
        logger.info('Disconnected.')

    def run(self):

        try:
            bars = self.make_bars()
            self.ib.run()
        except KeyboardInterrupt:
            self.disconnect(msg='KeyboardInterrupt')
        except Exception as errmsg:
            self.disconnect(msg=errmsg)
        else:
            logger.info('Doing nothing, end of run.')


if __name__ == "__main__":
    args = make_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    exchange = 'NYMEX'
    num_decimals = 2
    min_tick_size = 0.01

    s = StrategyTestReconnect(symbol=args.symbol,
                              contract_month=args.month,
                              exchange=exchange,
                              database=None,
                              num_decimals=num_decimals,
                              min_tick_size=min_tick_size,
                              client_id=args.client)
    logger.info('Trading {}'.format(s.contract))
    s.connect()
    s.run()
