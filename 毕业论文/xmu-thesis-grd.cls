%%=============================================================================
%% xmu-thesis-grd.cls for Xiamen University Graduate Thesis
%% modified by Maozhong Fu
%% Version: 1.2
%% Last update: 2021/12/28
%% Based on BIT-thesis at https://github.com/BIT-thesis/LaTeX-template
%%
%% 模块化版本 - 由 tex/ 目录下的模块文件组成
%%=============================================================================

%% 基础数学宏包（必须在最前面加载）
\RequirePackage{amsmath,amsthm,amsfonts,amssymb,bm,mathrsfs}
\RequirePackage{upgreek} % 直立希腊字母字体
\RequirePackage[final]{pdfpages}
\RequirePackage{ragged2e}
\RequirePackage{tabularx}

%% 获取当前目录（用于定位模块文件）
\def\xmu@texdir{tex/}

%% 加载模块文件
\input{\xmu@texdir 01-options.tex}    % 模板选项定义
\input{\xmu@texdir 02-labels.tex}     % 标签和默认内容
\input{\xmu@texdir 03-fonts.tex}      % 字体配置
\input{\xmu@texdir 04-layout.tex}     % 页面布局
\input{\xmu@texdir 05-chapters.tex}   % 章节标题格式
\input{\xmu@texdir 06-mainmatter.tex} % 前置/正文/后置设置
\input{\xmu@texdir 07-math.tex}       % 数学公式和定理
\input{\xmu@texdir 08-floats.tex}     % 图表和浮动体
\input{\xmu@texdir 09-hyperlink.tex}  % 超链接和目录样式
\input{\xmu@texdir 10-header-footer.tex} % 页眉页脚和脚注
\input{\xmu@texdir 11-cover.tex}      % 封面和声明
\input{\xmu@texdir 12-abstract.tex}   % 摘要和关键词
\input{\xmu@texdir 13-contents.tex}   % 中英文目录
\input{\xmu@texdir 14-lists.tex}      % 缩略词表/符号表/索引
\input{\xmu@texdir 15-bibliography.tex} % 参考文献
\input{\xmu@texdir 16-backmatter.tex} % 学术成果/致谢/附录
\input{\xmu@texdir 17-namelist.tex}   % 答辩委员会名单

%% 模板版本命令
\newcommand{\version}{\xmu@value@templateversion}

\endinput
