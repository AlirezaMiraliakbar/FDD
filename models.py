import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tqdm.notebook import trange, tqdm
from joblib import Parallel, delayed

from sklearn.decomposition import PCA
from scipy.stats import f

class QUANTS_module:
    def __init__(self, k, r, p, d, ARL0):
        """
        Initializes the SPC module class based on QUANTS method
        """
        self.k = k
        self.r = r
        self.ARL0 = ARL0
        self.d = d
        self.p = p

    def quantile_calc(self,ic_data):
        """
        For each data stream, gives quantiles using Phase I data
    
        Parameters
        ic_data : ic data for Phase I analysis
    
        Returns
        
        out : ic data quantiles
    
        """
        
        # print(f'Number of variables: {self.p}')
        self.quantiles = np.zeros([self.p, self.d - 1])
        for i in range(self.p):
            self.quantiles[i, :] = np.quantile(ic_data[i, :], np.arange(1 / self.d, 1, 1 / self.d))
    
        # return quantiles
    
    def fit(self, ic_list, h_ub, h_lb,eps1=10,eps2=0.001,max_iter=100):
        """
        Fitting the model, meaning to find the desired threshold based on a k value and other parameters

        Parameters:
        ic_list: Historical in-control data list
        quantiles: Calculated quantiles 
        h_ub: upper bound of h for bisection method
        h_lb: lower bound of h for bisection method
        eps1: threshold for ARL0 convergence (abs(ARL0-cal_ARL0) < eps1)
        eps2: threshold for h convergence (abs(h_ub - h_lb) < eps2)
        max_iter: maximum number of iterations for convergence

        Returns:
        h: the found threshold for the set of parameters
        """
        num_cores = multiprocessing.cpu_count()
        print(f"Number of CPU cores: {num_cores}")
        
        i = 0

        ARL0i = 0 # for intializing the while loop

        while abs(ARL0i - self.ARL0) > eps1 and abs(h_lb - h_ub) > eps2 and i < max_iter:
            h = 0.5 * (h_ub + h_lb)

            ARL0i, std_ARL0i, RL0_list = self.get_ARL(ic_list, h)
            
            if ARL0i < self.ARL0:
                h_lb = h
            if ARL0i > self.ARL0:
                h_ub = h
            
            i += 1

        self.h = h
        self.RL0_ = RL0_list

        return h, ARL0i, std_ARL0i, RL0_list
    
    def monitor(self, data, mode="online"):
        # something about q must be fixed I guess, maybe not!
        """
        Monitors new data points and checks whether they fall within the control limits.
        
        Parameters:
        data (float): The new data point to monitor.
        
        Returns:
        str: Whether the new data point is 'In control' or 'Out of control'.
        """
        # quantiles = self.quantile_calc(data)

        if np.isnan(data).any() or np.isnan(self.quantiles).any():
            raise ValueError("Data or quantiles contain NaN values.")

        q = data.shape[1]
           
        # print('I should not run more than one!')
        S1_up = np.zeros([self.d - 1, self.p])
        S2_up = np.zeros([self.d - 1, self.p])
        S1_down = np.zeros([self.d - 1, self.p])
        S2_down = np.zeros([self.d - 1, self.p])
        Sdif_down = np.zeros([self.d - 1, self.p])
        Sdif_up = np.zeros([self.d - 1, self.p])

        C_up = np.zeros(self.p)
        C_down = np.zeros(self.p)

        g_up = np.zeros(self.d - 1)
        g_down = np.zeros(self.d - 1)
        for l in range(self.d - 1):
            g_up[l] = 1 - (l + 1) / self.d
            g_down[l] = (l + 1) / self.d

        w_up = np.zeros([self.p, q])
        w_down = np.zeros([self.p, q])
        w = np.zeros([self.p, q])
        sum_topw = np.zeros(q)
        sum_topw_ranked = np.zeros(q)
        sum_topr = 0
        index_sorted = [[0 for x in range(self.p)] for y in range(q)]
        i = 0

        # i: time index
        # j: variable index

        num_false_alarm = 0
        while sum_topr < self.h and i < q:
            # print(f'i={i}')
            for j in range(self.p):
                A_up = np.zeros(self.d - 1)
                A_down = np.zeros(self.d - 1)

                Y = self.quantile_lookup(data[j, i], self.quantiles[j,:])
                quantile_index = next((i for i, x in enumerate(Y) if x), None)

                A_down[quantile_index:] = 1
                A_up = np.ones(self.d - 1) - A_down

                inter_up = Sdif_up[:, j] + A_up - g_up
                inter_down = Sdif_down[:, j] + A_down - g_down

                C_up[j] = np.matmul(np.matmul(inter_up, np.diag(1. / (g_up + S2_up[:, j]))), inter_up.transpose())
                C_down[j] = np.matmul(np.matmul(inter_down, np.diag(1. / (g_down + S2_down[:, j]))), inter_down.transpose())

                if C_up[j] > self.k:
                    S1_up[:, j] = (S1_up[:, j] + A_up) * (C_up[j] - self.k) / C_up[j]
                    S2_up[:, j] = (S2_up[:, j] + g_up) * (C_up[j] - self.k) / C_up[j]
                    Sdif_up[:, j] = S1_up[:, j] - S2_up[:, j]
                else:
                    S1_up[:, j] = 0
                    S2_up[:, j] = 0
                    Sdif_up[:, j] = 0

                w_up[j, i] = max(0, C_up[j] - self.k)

                if C_down[j] > self.k:
                    S1_down[:, j] = (S1_down[:, j] + A_down) * (C_down[j] - self.k) / C_down[j]
                    S2_down[:, j] = (S2_down[:, j] + g_down) * (C_down[j] - self.k) / C_down[j]
                    Sdif_down[:, j] = S1_down[:, j] - S2_down[:, j]
                else:
                    S1_down[:, j] = 0
                    S2_down[:, j] = 0
                    Sdif_down[:, j] = 0

                w_down[j, i] = max(0, C_down[j] - self.k)

                w[j, i] = max(w_up[j, i], w_down[j, i])

            sorted_list = np.sort(w[:, i])[::-1]
            index_sorted[i] = w[:, i].argsort()[::-1][:self.p]

            sum_topr = np.sum(sorted_list[:self.r])
            sum_topw[i] = sum_topr
            i += 1
            if mode == 'online':
                if i < 1206 and sum_topr > self.h: 
                    # print(f'false alarm at timestep = {i}')
                    S1_up = np.zeros([self.d - 1, self.p])
                    S2_up = np.zeros([self.d - 1, self.p])
                    S1_down = np.zeros([self.d - 1, self.p])
                    S2_down = np.zeros([self.d - 1, self.p])
                    Sdif_down = np.zeros([self.d - 1, self.p])
                    Sdif_up = np.zeros([self.d - 1, self.p])
            
                    C_up = np.zeros(self.p)
                    C_down = np.zeros(self.p)
            
                    g_up = np.zeros(self.d - 1)
                    g_down = np.zeros(self.d - 1)
                    for l in range(self.d - 1):
                        g_up[l] = 1 - (l + 1) / self.d
                        g_down[l] = (l + 1) / self.d
            
                    w_up = np.zeros([self.p, q])
                    w_down = np.zeros([self.p, q])
                    w = np.zeros([self.p, q])
                    # sum_topw = np.zeros(q)
                    sum_topr = 0
                    index_sorted = [[0 for x in range(self.p)] for y in range(q)]
                    num_false_alarm += 1
                
                
        return i-1, sum_topw[:i-1], num_false_alarm
        
    def get_ARL(self,data_list, h):

        """
    
        setting up parallel computing environment for calculating ARL
    
        Parameters
        ----------
        data : data stream for analysis
        h: constant threshold for top-r approach
        quantile_result: quantile vector

        Returns
        -------
        out : ARL and its standard deviation
    
        """
        T = len(data_list)
        num_cores = multiprocessing.cpu_count()
        # print(f"Number of CPU cores: {num_cores}")
        # logger.debug('get_ARL is called properly for Phase I analysis')
        Y = Parallel(n_jobs = num_cores - 1,max_nbytes= None)(delayed(self.quants)(data_list[i].T, h) for i in tqdm(range(T), desc='calculating RL0'))
        YY = [Y[i][0] for i in range(T)]
        Y1 = np.asarray(YY)
        ARL = np.mean(Y1)
        std_ARL = np.std(Y1)/np.sqrt(len(Y1))  
        print(f'ARL = {ARL} and std_ARL = {std_ARL} for h = {h} and k = {self.k}')
        
        return ARL, std_ARL, Y1
        
    def quantile_lookup(self,data_point, quantile_var):

        """
        Given data point, tells the quantile in which it resides
    
        Parameters
        ----------
        data_point: a single data point
        quantile_result: the in-control data quantiles obtained from quantile_calc
    
        Returns
        -------
        y: the indicator vector for the current observation
    
        """
        y = np.zeros([self.d])
        for i in range(self.d - 1):
            if data_point <= quantile_var[i]:
                y[i] = 1
                break
        if data_point > quantile_var[-1]:
            y[self.d - 1] = 1
    
        return y

    def quants(self,data, h):
        
        """
        The main algorithm that implements the QUANTS method

        Parameters
        ----------
        data: the observed measurements for Phase II analysis
        h: constant threshold to raise the alarm, related to the pre-specified in-control ARL

        Returns
        -------
        i-1: Out-of-control run length
        sum_topw[:i-1]: The top-r statistics sum for the observations
        """
        # in fitting, each data's each own quantiles are used
        quantiles = np.zeros([self.p, self.d - 1])
        for i in range(self.p):
            quantiles[i, :] = np.quantile(data[i, :], np.arange(1 / self.d, 1, 1 / self.d))

        if np.isnan(data).any() or np.isnan(quantiles).any():
            raise ValueError("Data or quantiles contain NaN values.")

        q = data.shape[1]
           
        
        S1_up = np.zeros([self.d - 1, self.p])
        S2_up = np.zeros([self.d - 1, self.p])
        S1_down = np.zeros([self.d - 1, self.p])
        S2_down = np.zeros([self.d - 1, self.p])
        Sdif_down = np.zeros([self.d - 1, self.p])
        Sdif_up = np.zeros([self.d - 1, self.p])

        C_up = np.zeros(self.p)
        C_down = np.zeros(self.p)

        g_up = np.zeros(self.d - 1)
        g_down = np.zeros(self.d - 1)
        for l in range(self.d - 1):
            g_up[l] = 1 - (l + 1) / self.d
            g_down[l] = (l + 1) / self.d

        w_up = np.zeros([self.p, q])
        w_down = np.zeros([self.p, q])
        w = np.zeros([self.p, q])
        sum_topw = np.zeros(q)
        sum_topw_ranked = np.zeros(q)
        sum_topr = 0
        index_sorted = [[0 for x in range(self.p)] for y in range(q)]
        i = 0
        # i: time index
        # j: variable index
        while sum_topr < h and i <= q - 1:
            for j in range(self.p):
                A_up = np.zeros(self.d - 1)
                A_down = np.zeros(self.d - 1)

                Y = self.quantile_lookup(data[j, i], quantiles[j,:])
                quantile_index = next((i for i, x in enumerate(Y) if x), None)

                A_down[quantile_index:] = 1
                A_up = np.ones(self.d - 1) - A_down

                inter_up = Sdif_up[:, j] + A_up - g_up
                inter_down = Sdif_down[:, j] + A_down - g_down

                C_up[j] = np.matmul(np.matmul(inter_up, np.diag(1. / (g_up + S2_up[:, j]))), inter_up.transpose())
                C_down[j] = np.matmul(np.matmul(inter_down, np.diag(1. / (g_down + S2_down[:, j]))), inter_down.transpose())

                if C_up[j] > self.k:
                    S1_up[:, j] = (S1_up[:, j] + A_up) * (C_up[j] - self.k) / C_up[j]
                    S2_up[:, j] = (S2_up[:, j] + g_up) * (C_up[j] - self.k) / C_up[j]
                    Sdif_up[:, j] = S1_up[:, j] - S2_up[:, j]
                else:
                    S1_up[:, j] = 0
                    S2_up[:, j] = 0
                    Sdif_up[:, j] = 0

                w_up[j, i] = max(0, C_up[j] - self.k)

                if C_down[j] > self.k:
                    S1_down[:, j] = (S1_down[:, j] + A_down) * (C_down[j] - self.k) / C_down[j]
                    S2_down[:, j] = (S2_down[:, j] + g_down) * (C_down[j] - self.k) / C_down[j]
                    Sdif_down[:, j] = S1_down[:, j] - S2_down[:, j]
                else:
                    S1_down[:, j] = 0
                    S2_down[:, j] = 0
                    Sdif_down[:, j] = 0

                w_down[j, i] = max(0, C_down[j] - self.k)

                w[j, i] = max(w_up[j, i], w_down[j, i])

            sorted_list = np.sort(w[:, i])[::-1]
            index_sorted[i] = w[:, i].argsort()[::-1][:self.p]

            sum_topr = np.sum(sorted_list[:self.r])
            sum_topw[i] = sum_topr

            i = i + 1

        return i - 1, sum_topw[:i - 1]
        
    def plot_histogram(self,array, bins=10):
        """
        Plots a histogram for a given NumPy array.
        
        Parameters:
        array (numpy.ndarray): The input array for the histogram.
        bins (int): Number of bins for the histogram (default: 10).
        """
        plt.hist(array, bins=bins, edgecolor='black')
        plt.title("Histogram of Calculated RL0")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
        plt.save('arl0_histogram.jpg')
         



class CDF_module():
        def __init__(self, k, r, ARL0, hist_data):
            """
            Initializes the SPC module class based on CDF method
            """
            self.k = k
            self.r = r
            self.ARL0 = ARL0
            self.hist_data_sorted = np.sort(hist_data,axis=0)

        def fit(self, ic_list, h_ub, h_lb, eps1=10, eps2=0.001, max_iter=100):

            """
            Fitting the model, meaning to find the desired threshold based on a k value and other parameters

            Parameters:
            ic_list: Historical in-control data list
            quantiles: Calculated quantiles 
            h_ub: upper bound of h for bisection method
            h_lb: lower bound of h for bisection method
            eps1: threshold for ARL0 convergence (abs(ARL0-cal_ARL0) < eps1)
            eps2: threshold for h convergence (abs(h_ub - h_lb) < eps2)
            max_iter: maximum number of iterations for convergence

            Returns:
            h: the found threshold for the set of parameters
            """

            num_cores = multiprocessing.cpu_count()
            print(f"Number of CPU cores: {num_cores}")
            
            i = 0

            ARL0i = 0 # for intializing the while loop

            while abs(ARL0i - self.ARL0) > eps1 and abs(h_lb - h_ub) > eps2 and i < max_iter:
                h = 0.5 * (h_ub + h_lb)

                ARL0i, std_ARL0i, RL0_list = self.get_ARL(ic_list, h)
                
                if ARL0i < self.ARL0:
                    h_lb = h
                if ARL0i > self.ARL0:
                    h_ub = h
                
                i += 1

            self.h = h
            self.RL0_ = RL0_list

            return h, ARL0i, std_ARL0i, RL0_list
        
        def monitor(self):
            """
            Monitors new data points and checks whether they fall within the control limits.
            
            Parameters:
            new_data (float): The new data point to monitor.
            
            Returns:
            Runlength as the total number of timesteps that raises the alarm
            """
            n_dst = data.shape[1]
            n_l = data.shape[0]
            max_RL = n_l
            W_plus = np.zeros([n_dst])
            W_minus = np.zeros([n_dst])
            W = np.zeros([n_dst])
            sum_topw = np.zeros(max_RL)
            sum_topr = 0
            RL = 0
            num_false_alarms = 0
            while sum_topr <= h and RL < max_RL:
                
                for j in range(n_dst):
                    
                    eta = self.bayes_cdf(self.hist_data_sorted[:,j], data[RL,j])
                    w_ = np.log(1 - eta)
                    w_p = np.log(eta)

                    # W plus/minus are based on time, not the data stream-> I mean we need historical data of W plus/minus
                    W_plus[j] = max(W_plus[j] - w_ - self.k, 0)
                    W_minus[j] = max(W_minus[j] - w_p - self.k, 0)
                    # we need to make a new row for the W_plus and minus
                    W[j] = max(W_plus[j], W_minus[j])
                
                sorted_list = np.sort(W[:])[::-1]
                sum_topr = np.sum(sorted_list[:self.r])
                sum_topw[RL] = sum_topr

                if mode == 'online' and RL < max_RL and sum_topr > h:

                    W_plus = np.zeros([n_dst])
                    W_minus = np.zeros([n_dst])
                    W = np.zeros([n_dst])
                    sum_topr = 0
                    num_false_alarms += 1

                RL += 1
        
            return RL - 1, sum_topw[:RL - 1], num_false_alarms
            
            
        def get_ARL(self,data_list, h):

            """
        
            setting up parallel computing environment for calculating ARL
        
            Parameters
            ----------
            data : data stream for analysis
            h: constant threshold for top-r approach
            quantile_result: quantile vector

            Returns
            -------
            out : ARL and its standard deviation
        
            """
            T = len(data_list)
            num_cores = multiprocessing.cpu_count()
            
            Y = Parallel(n_jobs = num_cores - 1,max_nbytes= None)(delayed(self.cdf)(data_list[i], h) for i in tqdm(range(T), desc='calculating RL0'))
            YY = [Y[i][0] for i in range(T)]
            Y1 = np.asarray(YY)
            ARL = np.mean(Y1)
            std_ARL = np.std(Y1)/np.sqrt(len(Y1))  
            print(f'ARL = {ARL} and std_ARL = {std_ARL} for h = {h} and k = {self.k}')
            
            return ARL, std_ARL, yY1
            
        def cdf(self,data, h):
            
            """
            The main algorithm that implements the CDF method

            Parameters
            ----------
            data: the observed measurements for Phase II analysis
            h: constant threshold to raise the alarm, related to the pre-specified in-control ARL

            Returns
            -------
            i-1: Out-of-control run length
            sum_topw[:i-1]: The top-r statistics sum for the observations
            """

            n_dst = data.shape[1]
            n_l = data.shape[0]
            max_RL = n_l
            W_plus = np.zeros([n_dst])
            W_minus = np.zeros([n_dst])
            W = np.zeros([n_dst])
            # Ws = np.zeros([n_dst])
            sum_topw = np.zeros(max_RL)
            sum_topr = 0
            RL = 0
            # z = np.zeros([n_l,n_dst])
            
            while sum_topr <= h and RL < max_RL:
                
                for j in range(n_dst):
                    
                    eta = self.bayes_cdf(self.hist_data_sorted[:,j], data[RL,j])
                    w_ = np.log(1 - eta)
                    w_p = np.log(eta)

                    # W plus/minus are based on time, not the data stream-> I mean we need historical data of W plus/minus
                    W_plus[j] = max(W_plus[j] - w_ - self.k, 0)
                    W_minus[j] = max(W_minus[j] - w_p - self.k, 0)
                    # we need to make a new row for the W_plus and minus
                    W[j] = max(W_plus[j], W_minus[j])


                sorted_list = np.sort(W[:])[::-1]
                sum_topr = np.sum(sorted_list[:self.r])
                sum_topw[RL] = sum_topr
                
                RL += 1
        
            return RL - 1, sum_topw[:RL - 1]
            
        def bayes_cdf(self,x_list_sort, x):
            """
            This function estimates the cdf of continuous and discrete distributions.
            Input:
                x_list_sort: sorted list
                x: new sample
            Output:
                Bayesian estimation of the cdf
            """
            if np.ndim(x_list_sort) == 1:
                return (np.searchsorted(x_list_sort, x) + 1)/(len(x_list_sort)+2)

            m, n = x_list_sort.shape
            cdf = np.zeros(m)
            for i in range(m):
                cdf[i] = np.searchsorted(x_list_sort[i], x[i])
            return (cdf+1)/(n+2) 
        
        def plot_histogram(self,array, bins=10):
            """
            Plots a histogram for a given NumPy array.
            
            Parameters:
            array (numpy.ndarray): The input array for the histogram.
            bins (int): Number of bins for the histogram (default: 10).
            """
            plt.hist(array, bins=bins, edgecolor='black')
            plt.title("Histogram of Calculated RL0")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()
#----------------------------------------------------------------------- PCA ------------------------------------------------------------------------

class PCA_T2():
    def __init__(self,alpha, variance_threshold):
        self.h_var = variance_threshold
        self.alpha = alpha

    def analysis(self, ic_train):
        pca_model = PCA()
        pca.fit(ic_train)
        # Calculate cumulative explained variance ratio
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        # Find the number of components that exceed the variance threshold
        num_components = np.argmax(cumulative_variance_ratio >= self.h_var) + 1
        self.k = num_components

    def fit(self, ic_train):
        
        self.pca_model = PCA(n_components = self.k)
        self.pca_model.fit(ic_train)
        
        # number of variables
        self.m = ic_train.shape[1]
        # number of samples
        self.n = ic_train.shape[0]
        
        coeff = m*(n-1)*(n+1)/(n*(n-m))

        #Get the critical value from the F-distribution
        F_critical = f.ppf(1 - self.alpha, self.m, self.n - self.m)

        self.T2_limit = coeff * F_critical

    def monitor(self, online_data, mode):
        
        max_RL = online_data[0]
        T2 = 0
        RL = 0
        while T2 <= self.T2_limit and RL < max_RL:
            on_projected = self.pca_model.transform(online_data[RL])
            # T² = scores * inverse(eigenvalues of PCA) * scores.T
            eigenvalues = self.pca_model.explained_variance_
            T2 = np.sum((on_projected ** 2) / eigenvalues, axis=1)
            num_false_alarms = 0
            if mode == 'online' and RL < max_RL and T2 > self.T2_limit:
                T2 = 0
                num_false_alarms += 1

            RL += 1
        
        return RL - 1, num_false alarms



        
        
           
    
    
