**另外一种方法是对训练基中的特征维度进行操作的，这次不是给每一个基分类器全部的特征，而是给不同的基分类器分不同的特征，即比如基分类器1训练前半部分特征，基分类器2训练后半部分特征（可以通过sklearn 的pipelines 实现）。最终通过StackingClassifier组合起来。**

============================

from sklearn.datasets import load_iris

from mlxtend.classifier import StackingClassifier

from mlxtend.feature_selection import ColumnSelector

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression

iris = load_iris() 

X = iris.data

y = iris.target

pipe1 = make_pipeline(ColumnSelector(cols=(0, 2)), LogisticRegression())

pipe2 = make_pipeline(ColumnSelector(cols=(1, 2, 3)),
                      LogisticRegression())
                      
sclf = StackingClassifier(classifiers=[pipe1, pipe2], 
                          meta_classifier=LogisticRegression())

sclf.fit(X, y)

============================================

StackingClassifier 使用API及参数解析：

StackingClassifier(classifiers, meta_classifier, use_probas=False, average_probas=False, verbose=0, use_features_in_secondary=False)
######
###### 参数：
#####
##### classifiers : 基分类器，数组形式，[cl1, cl2, cl3]. 每个基分类器的属性被存储在类属性 self.clfs_.
#####meta_classifier : 目标分类器，即将前面分类器合起来的分类器
#####use_probas : bool (default: False) ，如果设置为True， 那么目标分类器的输入就是前面分类输出的类别概率值而不是类别标签
#####average_probas : bool (default: False)，用来设置上一个参数当使用概率值输出的时候是否使用平均值。
#####verbose : int, optional (default=0)。用来控制使用过程中的日志输出，当 verbose = 0时，什么也不输出， verbose = 1，输出回归器的序号和名字。verbose = 2，输出详细的参数信息。verbose > 2, 自动将verbose设置为小于2的，verbose -2.
#####use_features_in_secondary : bool (default: False). 如果设置为True，那么最终的目标分类器就被基分类器产生的数据和最初的数据集同时训练。如果设置为False，最终的分类器只会使用基分类器产生的数据训练。

#####属性：
#####clfs_ : 每个基分类器的属性，list, shape 为 [n_classifiers]。
#####meta_clf_ : 最终目标分类器的属性

#####方法：

#####fit(X, y)
#####fit_transform(X, y=None, fit_params)
#####get_params(deep=True)，如果是使用sklearn的GridSearch方法，那么返回分类器的各项参数。
#####predict(X)
#####predict_proba(X)
#####score(X, y, sample_weight=None)， 对于给定数据集和给定label，返回评价accuracy
#####set_params(params)，设置分类器的参数，params的设置方法和sklearn的格式一样