# 1、介绍

1、Python：仅仅就是一门语言。

2、Python解释器

    python》C》汇编》计算机

3、pip

    pip就是一个很大的应用商城（都是python写的，且都是xxx.py)

4、Anaconda

    Python解释器+pip+一大堆用python写的程序（数据处理用的程序），帮你内置了很多数据处理的程序。

    还有一个虚拟环境管理

5、Pycharm和Jupyter

    pycharm内置了python解释器，并且具有文本编辑功能，还有各种各样的功能。
    python就是一个内置了python解释器的文本编辑器

    Jupyter同理

    推荐使用pycharm(社区版)来写代码

6、TensorFlow和 Pytorch ！！！！

    就是同级别的两个框架！(机器学习)
    
# 2、安装


这里安装的GPU版本的TensorFlow与Pytorch，需要先知道自己的电脑是否支持GPU版本的，不然只能安装cpu版本的。

查看是否支持GPU版本：
    计算机右击》管理》设备管理器》显示适配器->
    一般英伟达(NVIDIA),支持GPU；AMD显卡不支持GPU

    再去英伟达官网查看，自己的GPU适合哪个版本的CUDA，方便后面的安装。


1）安装Anaconda

    直接进入官网下载（安装最新的即可）
    https://www.anaconda.com/

    （最好以管理员身份)运行安装包.exe》选择next》选择I Agree》选择next》选择Just me或All users根据自己情况，其实都行》安装路径可以更改，》有两个选项，第一个是添加环境变量（add  Anaconda...PATH environment..)，第二个要勾选。
    在上上一步选择了Just me，添加环境变量选项可以勾选，你也可以在安装完后自行上网查看怎么手动添加环境变量；如果在上上步选择了All users，这里的第一个选项就不能勾选。。选择install》选择next》取消这里的两项勾选，直接选择Finish。

    安装完毕，打开Anaconda Prompt（这个会自行安装好，自己电脑开始菜单找找最近安装的文件），
    输入：conda -V
    验证是否安装成功，如果成功会显示版本号。
    (我看其他经验贴，他们能够直接打开命令提示符，输入：conda -V，，也能验证是否安装成功，，但我试了，我不能在这里验证，电脑会提示“conda不是内部或外部命令，也不是可运行的程序”，至于原因，我也不懂呀)

2）conda的一些命令

    conda可以用来创建虚拟环境，能够满足对不同版本Python的需要。

    比如创建虚拟环境：
        1、activate 激活
        2、conda create -n 环境名称  python=版本号
        3、conda activate  环境名称 ：激活这个环境
        4、conda deactivate ：退出这个环境

        5、删除虚拟环境
        6、conda remove -n your_env_name --all
            或：conda env remove --name your_env_name
        7、删除虚拟环境中的包：
            使用命令conda remove --name $your_env_name  $package_name（包名）
    其他conda命令：
        conda env list ：查看所有虚拟环境
        conda info -e  ：也是查看所有虚拟环境
        conda info或conda config --show 查看conda信息
    ---------------------------------------------------
        conda --version :查看conda版本
        conda update conda ：更新conda
        conda update Anaconda ：更新Anaconda确保稳定性与兼容性
    包：
        conda list：
        conda search 包名：查询是否有这个包
        conda install 包名 ：安装这个包
        conda uninstall 包名 ：卸载包
        conda clear -p --packages #删除没有用的包
    
3）修改conda创建虚拟环境的位置

    安装Anaconda后，创建新环境时环境自动安装在C盘，所以想要更改虚拟环境的位置；
    1、在conda info命令后，找到envs_dirs：下有路径，这就是创建虚拟环境的路径；
    2、conda config --add <想安装的环境路径>；：添加路径
       conda config --remove <环境路径> --all：删除路径
    还有一种方法：
        直接在C盘》用户》找到(.condarc)文件夹，记事本打开，在envs_dirs: 下直接添加新的环境路径。
    还有一种方法：
        查看目标路径的文件夹的权限：指定文件右键属性》安全》Users的权限》除特殊权限外，全部打钩》确定。

4）安装pycharm

    一般情况下，我们都要借助工具来辅助我们快速的搭建环境，编写代码以及运行程序。

        * IDE的概念
            IDE又称为集成开发环境，就是一款图形化界面的软件，它集成了编辑代码，编译代码，分析代码，执行代码以及调试代码的功能，在我们python开发中，最常用的IDE就是pycharm http://www.jetbrains.com/pycharm/download(下载社区版本 community，专业版花钱)
        
        * pycharm安装(社区版)
            * 双击安装

            * 自定义安装路径

            * 编辑设置(全部选中)

            * 开始的文件夹(默认就行)

            * 安装完成后双击
                如果是第一次使用的话选择不导入设置
                设置ui主题，喜欢就好
                下载一些配置可以下可以不下

            * 运行pycharm
                如果要新建项目，选择create new project，创建一个新的python工程
                如果要修改代码或者添加功能选择open

            * 创建文件的位置，并且点开下箭头
                new environment using表示创建一个项目就有一个单独的环境

                选择已存在的解释器(下面一个选项) interpreter
                选择路径，找到安装路径里python文件夹然后找到tools里的python.exe(如果找不到自己python安装在哪可以使用cmd的where python命令来查看路径，在windows环境下可能文件会被隐藏而IDE可能找不到文件夹，解决办法可以点击IDE上眼睛的标志用来显示隐藏文件夹)
                并且勾选上下面的选项，后面都使用这个环境
            
            * 创建新项目时在IDE文件夹上右键选择new --> python file --> 输入文件名，加不加后缀都行

            * 如果要改变文字大小
            File --> Settings --> Editor --> Font

            * 运行
            在代码写好后在空白处右键，Run (文件名) 即可

            * 下部终端可选择terminal，可在其中输入代码，效果和终端是一样的

            * 在实际的开发中可以对pycharm进行设置，在每次写项目的时候都会生成相应的内容(例如修改者之类的)
            File --> Settings --> File and Code Templates --> python script 在文件中添加想要的注释，例如
                ```
                时间 # @Time : ${DATE} ${TIME}
                作者 # @Author : lj
                文件名 # @File : ${NAME}
                项目名 # @Project : ${PROJECT_NAME}

5、CUDA的安装

    CUDA官网里面下载安装包。

    CUDA我这里装的是10.1版本的，，这个不建议装过新的版本。

    安装包管理员身份运行》可以改路径(不建议更改路径》同意》自定义》不勾选Display Driver和Other components 选择下一步》下一步》这里三个需要选择的安装位置，需要你在电脑里建一个文件夹，文件夹里面新建两个文件（例cuda1、cuda2），第一个和第二个安装路径就放在cuda1文件下，第三个安装路径就放在cuda2文件下，不要搞乱了！》下一步》这里的两个都不勾选，直接选择关闭。

    在命令提示符，输入：nvcc -V
    查看版本，没出错则安装成功。

6、CUDnn安装

    直接进入官网，CUDnn下载版本需要对应刚刚下载的CUDA10.1版本！

    下载完后：解压安装包》有三个文件夹：bin、include、lib》复制这三个文件夹，复制到刚刚cuda1的文件中，不会存在覆盖问题。就安装好了。

7、TensorFlow安装

    我们可以新建一个虚拟环境（tf2）：

    conda create -n tf2 python=3.7
    conda activate tf2
     我们用pip安装(用镜像安装)：
     pip install tensorflow_gpu==2.1.0 -i https://pypi.douban.com/simple --trusted-host pypi.douban.com
     直接回车，等待安装。

     上网找一段验证TensorFlow安装成功的代码》进入pycharm，新建项目，粘贴代码，不能直接运行，因为还需要配置解释器(配置刚刚创建的tf2).

     配置过程：文件》设置》project 项目名》python interpreter》在右边新的界面，找到(python interpreter)并选择show all，在新窗口点击+号》选择已经存在的环境，在配置好刚刚新建tf2的路径，找到tf2中的Tools》python.exe文件，选择这个》选择ok，等待导入包，带入成功后，就进入了tf2环境，并能够成功运行验证代码。

8、pytorch的安装

    官网，适配CUDA10.1的安装命令。









    
