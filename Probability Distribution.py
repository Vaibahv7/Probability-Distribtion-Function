#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # PMF

# In[2]:


l=[]
for i in range(10000):
    l.append(random.randint(1,6))


# In[3]:


l[:5]


# In[10]:


pb=(pd.Series(l).value_counts()/pd.Series(l).value_counts().sum()).sort_index()


# In[12]:


# PMF of 1 random variable
pb.plot(kind='bar')


# # For 2 Dice

# In[14]:


l=[]
for i in range(10000):
    a=random.randint(1,6)
    b=random.randint(1,6)
    l.append(a+b)


# In[16]:


pd.Series(l)


# In[21]:


pb=(pd.Series(l).value_counts()/pd.Series(l).value_counts().sum()).sort_index()


# In[23]:


pb


# In[22]:


pb.plot(kind='bar')


# # CDF

# In[24]:


import numpy as np
np.cumsum(pb)


# In[25]:


np.cumsum(pb).plot(kind='bar')


# It says that prob of x<=5 is 0.277,, for x<=2 it is 0.028

# # Probability Density Function(Parametric-assumption based)

# In[26]:


from numpy.random import normal
sample=normal(loc=50,scale=5,size=1000)


# In[33]:


#sample prob distribution
plt.hist(sample,bins=10)


# We can see that it is somewhat normal type

# In[34]:


sample_mean=sample.mean()
sample_std=sample.std()


# In[36]:


sample_std


# In[38]:


from scipy.stats import norm


# In[39]:


#Input sample mean ,std in Normal Distribution as we can see from sample prob dist that it follows normal dist
dist=norm(sample_mean,sample_std)


# In[40]:


# provide x values to noramal dist function
values=np.linspace(sample.min(),sample.max(),100)


# In[41]:


prob_density=[dist.pdf(values) for x in values]


# In[47]:


plt.hist(sample,bins=10)
plt.plot(values,prob_density)


# In[48]:


import seaborn as sns
sns.distplot(sample)


# # Probability Desntiy Function(Non-Parametric- estimates based on data)

# KDE

# In[50]:


s1=normal(loc=20,scale=5,size=300)
s2=normal(loc=40,scale=5,size=700)
sample=np.hstack((s1,s2))


# In[51]:


plt.hist(sample,bins=50)


# In[58]:


from sklearn.neighbors import KernelDensity
model=KernelDensity(bandwidth=2,kernel='gaussian')
sample=sample.reshape(len(sample),1)


# In[59]:


model.fit(sample)


# In[60]:


values=np.linspace(sample.min(),sample.max(),100)
values=values.reshape(len(values),1)


# In[61]:


prob_kde=model.score_samples(values)
prob_kde=np.exp(prob_kde)


# In[62]:


plt.hist(sample,bins=50,density=True)
plt.plot(values[:],prob_kde)
plt.show()


# In[65]:


sns.kdeplot(sample.reshape(1000),bw_adjust=.5)


# In[66]:


import seaborn as sns


# In[67]:


df=sns.load_dataset('iris')


# In[68]:


df.head()


# In[73]:


#plot KDE for sepal_length, sepal_width,sepal_widht,petal_lenght wrt species
sns.kdeplot(data=df,x='sepal_length',hue='species')


# In[74]:


sns.kdeplot(data=df,x='sepal_width',hue='species')


# In[75]:


sns.kdeplot(data=df,x='petal_length',hue='species')


# In[76]:


sns.kdeplot(data=df,x='petal_width',hue='species')


# We can infer that petal_length and petal_width are better features to predict the species as 
# 
# it shows that the probability distribution in case of petal_length and petal_width better categorise species than sepal_length and sepal_width 

# In[78]:


sns.kdeplot(data=df,x='petal_width',hue='species')
sns.ecdfplot(data=df,x='petal_width',hue='species')


# Here PDF shows that if  0.7< peta_width <1.7 then is it Versicolor
# 
# CDF shows the accuracy of PDF

# In[80]:


sns.jointplot(data=df,x='petal_length',y='sepal_length',kind='kde',fill=True,cbar=True)


# It is 2D PDF, it respresents the density of probability of combination of 2 features
# 
# Darker the plot(blue), higher the density

# # Convert Normal Distribution to Standard Normal Distribution(mean=0,std=1)

# In[81]:


df['sepal_length'].mean()


# In[82]:


df['sepal_length'].std()


# In[83]:


x=(df['sepal_length']-df['sepal_length'].mean())/df['sepal_length'].std()


# In[84]:


sns.kdeplot(x)


# We can see that the Normal distribution has changed to SND with mean=0 and std=1

# In[85]:


x.mean()


# In[86]:


x.std()


# # QQ PLOT

# In[87]:


sns.kdeplot(df['sepal_length'])


# In[89]:


tmp=sorted(df['sepal_length'].tolist())


# In[90]:


y_quant=[]
for i in range(1,101):
    y_quant.append(np.percentile(tmp,i))


# In[91]:


samples=np.random.normal(loc=0,scale=1,size=1000)


# In[93]:


x_quant=[]
for i in range(1,101):
    x_quant.append(np.percentile(samples,i))


# In[95]:


sns.scatterplot(x=x_quant,y=y_quant)


# In[96]:


import statsmodels.api as ap
import matplotlib.pyplot as plt


# In[98]:


fig=ap.qqplot(df['sepal_length'],line='45',fit=True)


# # Pareto Distribution

# In[99]:


alp=3
xm=1

x=np.linspace(0.1,10,1000)
y=alp*(xm**alp)/x**(alp+1)


# In[100]:


plt.plot(x,y)


# In[102]:


plt.plot(np.log(x),np.log(y))


# # Binomial Distribution

# In[106]:


n=10
p=0.8
size=1000

bin_dist=np.random.binomial(n,p,size)

plt.hist(bin_dist,density=True)

plt.show()


# In[107]:


# It reprents no of times head occurs when coin tossed 10 times 
bin_dist


# # One Sample t-Test

# In[135]:


t_train=pd.read_csv('t_train.csv').drop(columns=['Survived'])
t_test=pd.read_csv('t_test.csv')


# In[136]:


final=pd.concat([t_train,t_test]).sample(1309)


# In[137]:


pop=final['Age'].dropna()


# In[138]:


sam_age=pop.sample(25).values


# In[139]:


sam_age


# Ho- pop(mean)=35
# 
# H1- pop(mean)<35
# 
# Here we have sample size<=30, so acc to CLT if do not follow ND
# 
# So we apply Shapiro test to check whether our sample follows ND 

# In[140]:


from scipy.stats import shapiro


# In[141]:


shapiro.age=shapiro(sam_age)

print(shapiro.age)


# If pval<.05, then sample is not ND.
# 
# Since pvalue is >0.05, we can say that sample follows ND

# In[142]:


pop_mean =35


# In[143]:


import scipy.stats as sts

t_stats, p_value=sts.ttest_1samp(sam_age,pop_mean)

print('t_stats: ',t_stats)
print('p_value: ',p_value)


# In[144]:


aplha=0.05

if p_value<aplha:
    print('Reject Null Hypothesis(H0)')
else:
    print('Do not reject Null Hypothesis(H1)')


# We reject null Hypothesis which says that mean age of population is 35,
# 
# we can see it from mean of popuation that pop(mean)<35 which is 29.88

# In[145]:


pop.mean()


# # 2 Sample t-test(Independent Variables)

# In[163]:


desk_t=(list(np.random.randint(low = 10,high=25,size=30)))
mob_t=(list(np.random.randint(low = 10,high=20,size=30)))


# In[164]:


shapiro_desk=shapiro(desk_t)
shapiro_mob=shapiro(mob_t)

print('Sahpiro Wilk test for Desktop users: ',shapiro_desk)
print('Sahpiro Wilk test for Mobile users: ',shapiro_mob)


# In[165]:


from scipy.stats import levene

lev_test=levene(desk_t,mob_t)
print(lev_test)


# if for levene test p_val<0.05 then variance of both variable differs
# 
# Here we can see that p_val>0.05, hence variance of both variable is equal

# In[166]:


from scipy.stats import t

t_val=-5.25

df=58                       #(degree of freedom)

cdf_val=t.cdf(t_val,df)

print(cdf_val*2)


# As we can see that p_val < 0.5
# 
# so we reject null hypothesis which says that mean(desk)=mean(mob)

# # Paired t-test(dependent variable)

# In[180]:


before=np.array(np.random.randint(low = 68,high=92,size=15))
after=np.array(np.random.randint(low = 67,high=93,size=15))


# In[181]:


diff=before-after


# In[182]:


shapiro_tst=shapiro(diff)
print('Shapiro-Wilk Test: ',shapiro_tst)


# Since p_val > 0.05 we can say that the diff follows ND

# In[183]:


diff_mean=np.mean(diff)
diff_std=np.std(diff)


# In[184]:


n=len(diff)

t_stats=diff_mean/(diff_std/np.sqrt(n))
print(t_stats)


# In[185]:


df=n-1

p_val=sts.t.cdf(t_stats,df)

print(p_val)


# We can see that p_val > 0.05
# 
# Hence we fail to reject null Hypo which says that there is no reduction in weights after training prog
# 

# In[ ]:




