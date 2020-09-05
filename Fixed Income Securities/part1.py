import numpy as np
import pandas as pd
from enum import Enum
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from matplotlib import pyplot as plt


class underlying_type(Enum):
    BOND = 0
    LIBOR = 1
    OIS = 2


class frequency(Enum):
    quater = 0.25
    semi_annual = 0.5
    annual = 1


class IRS:
    def __init__(self, frequency, DF_fun):
        self.frequency = frequency
        self.discount_fun = DF_fun

    def add_market_data(self, data):
        self.__market_tenor = data[0]
        self.__market_rate = data[1]
        self.implied_underlying_DF_bootstrap = IRS.__implied_underlying_DF_bootstrap(
            self)  #添加数据自动运行一次，并储存为属性，供后续函数一次性使用，无需重复跑

    def __implied_underlying_DF_bootstrap(self):
        freq = self.frequency.value
        tenor = self.__market_tenor
        DF_fun = self.discount_fun
        if freq not in tenor:
            raise Exception(
                'algo needs intial underlying float spot rate with frequency')
        else:
            multi = tenor / freq
            check_int_multi = [
                multi[i] == round(multi[i]) for i in range(len(multi))
            ]
            int_multi_index = np.argwhere(check_int_multi).flatten()[
                1:]  # filter for useful data, delta整数倍的市场数据
            rate = self.__market_rate
            first_unsed_data_index = int(np.argwhere(tenor == freq))
            float_leg_rate_list = np.array([rate[first_unsed_data_index]])
            DF_list = np.array(
                [
                    float(DF_fun(0, freq * i))
                    for i in range(1, int(tenor[int_multi_index[-1]] / freq +
                                          1))
                ]
            )  # 未考虑溢出期限的情况 如 T30.5 freq0.5 多出半年归属问题也未考虑市场出现tenor小于freq的IRS rate
            DF_cum = np.cumsum(DF_list)
            for i in int_multi_index:  #应该找列表中delta倍数期限的产品bootstrap
                known_float_pay_num = len(float_leg_rate_list)
                total_payment_times = int(tenor[i] / freq)
                unknown_float_pay_num = total_payment_times - known_float_pay_num
                IRS_formula_part1 = 'rate[{i}]*DF_cum[{total_payment_times}]-np.sum(float_leg_rate_list*DF_list[:{known_float_pay_num}])'.format(
                    i=i,
                    total_payment_times=total_payment_times - 1,
                    known_float_pay_num=known_float_pay_num)
                IRS_formula_part2 = ''
                for j in range(unknown_float_pay_num):
                    IRS_formula_part2 += '-DF_list[{DF_list_index}]*x[{j}]'.format(
                        DF_list_index=known_float_pay_num + j, j=j)
                IRS_formula = IRS_formula_part1 + IRS_formula_part2
                linear_interpo_formula_basis = 'np.prod(1/(1+freq*float_leg_rate_list))'

                def f(x):
                    locals().update({
                        'freq': freq,
                        'rate': rate,
                        'DF_list': DF_list,
                        'DF_cum': DF_cum,
                        'float_leg_rate_list': float_leg_rate_list
                    })
                    y = [0] * unknown_float_pay_num
                    for k in range(unknown_float_pay_num):
                        linear_interpo_formula_part1 = '{linear_interpo_formula_basis}'.format(
                            linear_interpo_formula_basis=
                            linear_interpo_formula_basis)
                        linear_interpo_formula_part2 = '{linear_interpo_formula_basis}/(1+freq*x[0])/(1+freq*x[1])'.format(
                            linear_interpo_formula_basis=
                            linear_interpo_formula_basis)
                        linear_interpo_formula_part3 = '2*{linear_interpo_formula_basis}/(1+freq*x[0])'.format(
                            linear_interpo_formula_basis=
                            linear_interpo_formula_basis)
                        for l in range(k - 1):
                            linear_interpo_formula_part1 += '/(1+freq*x[{index}])'.format(
                                index=l)
                            linear_interpo_formula_part2 += '/(1+freq*x[{index}])'.format(
                                index=l + 2)
                            linear_interpo_formula_part3 += '/(1+freq*x[{index}])'.format(
                                index=l + 1)
                        linear_interpo_formula = linear_interpo_formula_part1 + '+' + linear_interpo_formula_part2 + '-' + linear_interpo_formula_part3
                        y[k] = eval(IRS_formula) if k == 0 else eval(
                            linear_interpo_formula)
                    return y

                float_leg_rate_list = np.append(
                    float_leg_rate_list, fsolve(f,
                                                [0] * unknown_float_pay_num))
            underlying_spot_DF_list = 1 / np.cumprod(1 + freq *
                                                     float_leg_rate_list)
            tenor_match_DF_list = tenor[tenor < freq]
            tenor_match_DF_list = np.append(
                tenor_match_DF_list,
                [freq * i for i in range(1,
                                         int(tenor[-1] / freq) + 1)])
            #DF_list未考虑市场出现tenor小于freq的IRS rate
            return tenor_match_DF_list, underlying_spot_DF_list

    def underlying_DF(self, A, B, *, kind='linear'):
        if A > B:
            raise Exception(
                'error in IRS.underlying_DF/rate(): B should greater than A')
        try:
            tenor_match_DF_list, DF_list = self.implied_underlying_DF_bootstrap
            f = interp1d(tenor_match_DF_list, DF_list, kind=kind)
            Lower = tenor_match_DF_list.min()
            Upper = tenor_match_DF_list.max()
            if Lower <= B <= Upper and A == 0:
                return f(B)
            elif Lower <= A <= Upper and Lower <= B <= Upper:
                return f(B) / f(A)
            else:
                print(
                    'cannot interpolate, A/B out of range, shall within [%s,%s]'
                    % (Lower, Upper))
        except Exception as e:
            print(e)

    def underlying_DF_curve(self, show=False):
        try:
            tenor_match_DF_list, DF_list = self.implied_underlying_DF_bootstrap
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(tenor_match_DF_list, DF_list)
            ax1.set_xlabel('tenor')
            ax1.set_ylabel('DF')
            ax1.set_title('LIBOR discount factor')
            if show:
                plt.show()
            return fig1
        except AttributeError:
            print('add_market_data first!')

    def underlying_rate(self, A, B):
        '''rate convention: L(A,B), e.g., L(1y,2y)(fwd); L(0,1y)(spot)'''
        return (1 / IRS.underlying_DF(self, A, B) - 1) / (B - A)

    def IRS_pricer(self, n, m):
        '''swap convetion: n*m, e.g.,1y*10y(fwd);0y*1y(spot)'''
        Lower = self.__market_tenor[0]
        Upper = self.__market_tenor[-1]
        if 0 < n < Lower:
            raise Exception('error in IRS.IRS_pricer(): n = 0 or n >= %s' %
                            Lower)
        if n + m > Upper:
            raise Exception('error in IRS.IRS_pricer(): n+m <= %s' % Upper)
        freq = self.frequency.value
        DF_spot_cal = self.discount_fun
        float_leg_rate_cal = IRS.underlying_rate
        settlement_num = max(int(m / freq), 1)
        numerator = sum([
            DF_spot_cal(0, n + min(freq, m) * (i + 1)) *
            float_leg_rate_cal(self, n + min(freq, m) * i, n + min(freq, m) *
                               (i + 1)) for i in range(settlement_num)
        ])
        denominator = sum([
            DF_spot_cal(0, n + min(freq, m) * (i + 1))
            for i in range(settlement_num)
        ])
        IRS_price = numerator / denominator
        return IRS_price


class OIS:
    def __init__(self, frequency):
        self.frequency = frequency

    def add_market_data(self, data):
        self.__market_tenor = data[0]
        self.__market_rate = data[1]
        self.__DF_bootstrap = OIS.__DF_bootstrap(self)
        #添加数据自动运行一次，并储存为属性，供后续函数一次性使用，无需重复跑

    def __DF_bootstrap(self):
        try:
            tenor = np.array(self.__market_tenor)
            rate = np.array(self.__market_rate)
            freq = self.frequency.value
            DF_list = np.array([])
            if freq not in tenor:
                raise Exception(
                    'algo needs starting OIS rate with tenor equal to frequency'
                )
            else:
                for i in range(len(tenor)):
                    if tenor[i] <= freq:
                        DF_list = np.append(DF_list,
                                            [1 / (1 + tenor[i] * rate[i])])
                    else:
                        unkown_var_num = int((tenor[i] - tenor[i - 1]) / freq)

                        def f(x):
                            y = [0] * unkown_var_num
                            if unkown_var_num == 1:
                                y[0] = (1 - x[0]) / (
                                    freq *
                                    (np.sum(DF_list[len(tenor[tenor < freq]):])
                                     + x[0])) - rate[i]
                            else:
                                y[0] = (1 - (x[0] * tenor[i] + x[1])) / (
                                    freq *
                                    (np.sum(DF_list[len(tenor[tenor < freq]):])
                                     + x[0] * 0.5 * unkown_var_num *
                                     (tenor[i - 1] + freq + tenor[i]) +
                                     x[1] * unkown_var_num)) - rate[i]
                                y[1] = x[0] * tenor[i - 1] + x[1] - DF_list[-1]
                            return y

                        result = fsolve(f, [0] * unkown_var_num)
                        DF_list = np.append(
                            DF_list,
                            result) if unkown_var_num == 1 else np.append(
                                DF_list, [
                                    result[0] *
                                    (tenor[i - 1] + freq * j) + result[1]
                                    for j in range(1, unkown_var_num + 1)
                                ])
                tenor_match_DF_list = tenor[tenor < freq]
                tenor_match_DF_list = np.append(
                    tenor_match_DF_list,
                    [freq * i for i in range(1,
                                             int(tenor[-1] / freq) + 1)])
            return tenor_match_DF_list, DF_list
        except AttributeError:
            print('add_market_data first!')

    def DF(self, A, B, *, kind='linear'):
        if A > B:
            raise Exception('error in OIS.DF(): B should greater than A')
        try:
            tenor_match_DF_list, DF_list = self.__DF_bootstrap
            f = interp1d(tenor_match_DF_list, DF_list, kind=kind)
            Lower = tenor_match_DF_list.min()
            Upper = tenor_match_DF_list.max()
            if Lower <= B <= Upper and A == 0:
                return f(B)
            elif Lower <= A <= Upper and Lower <= B <= Upper:
                return f(B) / f(A)
            else:
                print(
                    'cannot interpolate, A/B out of range, shall within [%s,%s]'
                    % (Lower, Upper))
        except Exception as e:
            print(e)

    def DF_curve(self, show=False):
        try:
            tenor_match_DF_list, DF_list = self.__DF_bootstrap
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(tenor_match_DF_list, DF_list)
            ax1.set_xlabel('tenor')
            ax1.set_ylabel('DF')
            ax1.set_title('OIS discount factor')
            if show:
                plt.show()
            return fig1
        except AttributeError:
            print('add_market_data first!')

    def OIS_pricer(self, n, m):
        '''swap convetion: n*m, e.g.,1y*10y(fwd);0y*1y(spot)'''
        Lower = self.__market_tenor[0]
        Upper = self.__market_tenor[-1]
        if 0 < n < Lower:
            raise Exception('error in OIS.OIS_pricer(): n = 0 or n >= %s' %
                            Upper)
        if n + m > Upper:
            raise Exception('error in OIS.OIS_pricer(): n+m <= %s' % Upper)
        freq = self.frequency.value
        DF_spot_cal = OIS.DF
        settlement_num = max(int(m / freq), 1)
        numerator = DF_spot_cal(self, 0, n) - DF_spot_cal(
            self, 0, n + m) if n != 0 else 1 - DF_spot_cal(self, 0, n + m)
        denominator = min(m, freq) * sum([
            DF_spot_cal(self, 0, n + min(freq, m) * (i + 1))
            for i in range(settlement_num)
        ])
        OIS_price = numerator / denominator
        return OIS_price


def data_processor(instance_var, data_path, sheet_name):
    '''retrieve data from excel to instance'''
    df = pd.read_excel(data_path,
                       sheet_name=sheet_name,
                       header=0,
                       index_col=None,
                       usecols=range(0, 3))
    data = df.loc[:, ['Tenor', 'Rate']].values.T
    for i in range(len(data[0])):
        data[0][i] = float(data[0][i][0:-1]) if data[0][i][-1] in [
            'Y', 'y'
        ] else (float(data[0][i][0:-1]) / 12)  # reconcile unit into year
        # to update algo, shall consider day count convension
    instance_var.add_market_data(data)


if __name__ == "__main__":

    data_path = r'IR Data.xlsx'
    OIS_instance = OIS(frequency.annual)
    IRS_instance = IRS(frequency.semi_annual, OIS_instance.DF)
    data_processor(OIS_instance, data_path, 'OIS')
    data_processor(IRS_instance, data_path, 'IRS')
    # part1.1
    OIS_instance.DF_curve(show=True)
    # part1.2
    IRS_instance.underlying_DF_curve(show=True)
    # part1.3
    data = [[IRS_instance.IRS_pricer(i, j) for j in [1, 2, 3, 5, 10]]
            for i in [1, 5, 10]]
    fwd_IRS_df = pd.DataFrame(data=data,
                              index=[1, 5, 10],
                              columns=[1, 2, 3, 5, 10])
    print(fwd_IRS_df)