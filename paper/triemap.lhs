% -*- latex -*-

% Links
% https://stackoverflow.com/questions/16084788/generic-trie-haskell-implementation
% Hinze paper: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.8.4069

%% For double-blind review submission, w/o CCS and ACM Reference (max submission space)
\documentclass[sigplan,review,dvipsnames]{acmart}\settopmatter{printfolios=true,printccs=false,printacmref=false}
%% For double-blind review submission, w/ CCS and ACM Reference
%\documentclass[sigplan,review,anonymous]{acmart}\settopmatter{printfolios=true}
%% For single-blind review submission, w/o CCS and ACM Reference (max submission space)
%\documentclass[sigplan,review]{acmart}\settopmatter{printfolios=true,printccs=false,printacmref=false}
%% For single-blind review submission, w/ CCS and ACM Reference
%\documentclass[sigplan,review]{acmart}\settopmatter{printfolios=true}
%% For final camera-ready submission, w/ required CCS and ACM Reference
%\documentclass[sigplan,dvipsnames]{acmart}\settopmatter{}


%% Journal information
%% Supplied to authors by publisher for camera-ready submission;
%% use defaults for review submission.
\acmJournal{PACMPL}
\acmVolume{1}
\acmNumber{ICFP} % CONF = POPL or ICFP or OOPSLA
\acmArticle{1}
\acmYear{2022}
\acmMonth{5}
\acmDOI{} % \acmDOI{10.1145/nnnnnnn.nnnnnnn}
\startPage{1}

%% Copyright information
%% Supplied to authors (based on authors' rights management selection;
%% see authors.acm.org) by publisher for camera-ready submission;
%% use 'none' for review submission.
\setcopyright{none}
%\setcopyright{acmcopyright}
%\setcopyright{acmlicensed}
%\setcopyright{rightsretained}
%\copyrightyear{2018}           %% If different from \acmYear

%% Bibliography style
\bibliographystyle{ACM-Reference-Format}
%% Citation style
%% Note: author/year citations are required for papers published as an
%% issue of PACMPL.
\citestyle{acmauthoryear}   %% For author/year citations


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Note: Authors migrating a paper from PACMPL format to traditional
%% SIGPLAN proceedings format must update the '\documentclass' and
%% topmatter commands above; see 'acmart-sigplanproc-template.tex'.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Some recommended packages.
\usepackage{booktabs}   %% For formal tables:
                        %% http://ctan.org/pkg/booktabs
\usepackage{subcaption} %% For complex figures with subfigures/subcaptions
                        %% http://ctan.org/pkg/subcaption
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{amsthm}
% \usepackage{amssymb}
% \usepackage{stmaryrd}
\usepackage{framed}
\usepackage{proof}
\usepackage{braket}
\usepackage{fancyvrb}
\usepackage{listings}
\usepackage[inline,shortlabels]{enumitem}
\usepackage[capitalize]{cleveref}
\usepackage{xcolor}
\usepackage{pgffor}
\usepackage{ragged2e}
\usepackage{upgreek} % \textmu
\usepackage{multirow}
\usepackage{diagbox}
\usepackage{csquotes}
\usepackage{comment}  % 'comment' environment

% \RequirePackage{xargs}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\pgfplotsset{compat=1.16}

\VerbatimFootnotes

\lstset{language=Haskell}

\let\restriction\relax

\theoremstyle{theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{theorem}{Theorem}
\newtheorem{corollary}{Corollary}
\newtheorem{property}{Property}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{restriction}{Restriction}
\newtheorem{intuition}{Intuition}
\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newtheorem{notation}{Notation}

\crefname{figure}{Fig.}{Figs.}
\Crefname{figure}{Fig.}{Figs.}
\crefname{restriction}{Restriction}{Restrictions}

% Tables should have the caption above
\floatstyle{plaintop}
\restylefloat{table}

\clubpenalty = 1000000
\widowpenalty = 1000000
\displaywidowpenalty = 1000000

% Some colors:
\definecolor{dkcyan}{rgb}{0.1, 0.3, 0.3}
\definecolor{dkgreen}{rgb}{0,0.3,0}
\definecolor{olive}{rgb}{0.5, 0.5, 0.0}
\definecolor{dkblue}{rgb}{0,0.1,0.5}

\definecolor{col:ln}{rgb}  {0.1, 0.1, 0.7}
\definecolor{col:str}{rgb} {0.8, 0.0, 0.0}
\definecolor{col:db}{rgb}  {0.9, 0.5, 0.0}
\definecolor{col:ours}{rgb}{0.0, 0.7, 0.0}

\definecolor{lightgreen}{RGB}{170, 255, 220}
\definecolor{darkbrown}{RGB}{121,37,0}

% Customized syntax highlighting for Haskell code snippets:
\colorlet{listing-comment}{gray}
\colorlet{operator-color}{darkbrown}

\lstdefinestyle{default}{
    basicstyle=\ttfamily\fontsize{8.7}{9.5}\selectfont,
    columns=fullflexible,
    commentstyle=\sffamily\color{black!50!white},
    escapechar=\#,
    framexleftmargin=1em,
    framexrightmargin=1ex,
    keepspaces=true,
    keywordstyle=\color{dkblue},
    mathescape,
    numbers=none,
    numberblanklines=false,
    numbersep=1.25em,
    numberstyle=\relscale{0.8}\color{gray}\ttfamily,
    showstringspaces=true,
    stepnumber=1,
    xleftmargin=1em
}

\lstdefinelanguage{custom-haskell}{
    language=Haskell,
    deletekeywords={lookup, delete, map, mapMaybe, Ord, Maybe, String, Just, Nothing, Int, Bool},
    keywordstyle=[2]\color{dkgreen},
    morekeywords=[2]{String, Map, Ord, Maybe, Int, Bool},
    morekeywords=[2]{Name, Expression, ESummary, PosTree, Structure, HashCode, VarMap},
    keywordstyle=[3]\color{dkcyan},
    mathescape=false, % so that we can write infix $
    escapechar=\%,    % ... but still have a way to write math between % %
    literate=%
        {=}{{{\color{operator-color}=}}}1
        {||}{{{\color{operator-color}||}}}1
        {\\}{{{\color{operator-color}\textbackslash$\,\!$}}}1
        {.}{{{\color{operator-color}.}}}1
        {=>}{{{\color{operator-color}=>}}}1
        {->}{{{\color{operator-color}->}}}1
        {<-}{{{\color{operator-color}<-}}}1
        {::}{{{\color{operator-color}::}}}1
}

\lstset{style=default}
% Environment for code snippets
\lstnewenvironment{code}[1][]
  {\small\lstset{language=custom-haskell,#1}}
  {}

% Environment for example expressions
\lstnewenvironment{expression}[1][]
  {\small\lstset{#1}}
  {}

%include rae.fmt
\newcommand{\keyword}[1]{\textcolor{BlueViolet}{\textbf{#1}}}
\newcommand{\id}[1]{\textsf{\textsl{#1}}}
\newcommand{\varid}[1]{\textcolor{Sepia}{\id{#1}}}
\newcommand{\conid}[1]{\textcolor{OliveGreen}{\id{#1}}}
\newcommand{\tick}{\text{\textquoteright}}
\newcommand{\package}[1]{\textsf{#1}}

% Change lhs2TeX code indentation
% https://tex.stackexchange.com/a/186520/52414
\setlength\mathindent{0em}
% https://tex.stackexchange.com/a/58131
\renewcommand{\hscodestyle}{\small}

%if style == poly
%format checktype(e) = e
%format property name (vars) (lhs) (rhs) = " " lhs "\equiv" rhs
%format propertyImpl name (vars) (premise) (lhs) (rhs) = premise "\Rightarrow" lhs "\equiv" rhs

% Abbreviations
%format lookupEM = lkEM
%format lookupTM = lkTM
%format lookupMM = lkMM
%format lookupMTM = lkMTM
%format lookupLM = lkLM
%format lookupLEM = lkLEM
%format lookupSEM = lkSEM
%format lookupMSEM = lkMSEM
%format alterEM = atEM
%format alterTM = atTM
%format alterMM = atMM
%format alterMTM = atMTM
%format alterLM = atLM
%format alterSEM = atSEM
%format alterMSEM = atMSEM
%format DeBruijnEnv = DBEnv
%format TrieKey = Key
%format MTrieKey = MKey
%format lookupPatMTM = lkMTM
%format alterPatMTM = atMTM
%format lookupPatMSEM = lkMSEM
%format alterPatMSEM = atMSEM
%format e1
%format e2
%format m1
%format m2
%format z1
%format v1
%format v2
%format var1
%format var2
%format app1
%format app2
%endif

%if style == newcode
%format property name (vars) (lhs) (rhs) = name " = \ " vars " -> (" lhs ") == (" rhs ")"
%format propertyImpl name (vars) (premise) (lhs) (rhs) = name "= \ " vars " -> (" premise ") ==> (" lhs ") == (" rhs ")"

\begin{code}
{-# LANGUAGE ScopedTypeVariables, TypeFamilies, AllowAmbiguousTypes,
             TemplateHaskell, ExtendedDefaultRules, DataKinds, NoMonomorphismRestriction #-}
{-# OPTIONS_GHC -Wall -Werror -Wno-missing-signatures -Wno-type-defaults -Wno-orphans
                -Wno-unused-matches #-}
{-# OPTIONS_GHC -Wwarn=incomplete-patterns #-}

module TrieMap where

import Prelude (String, Int, Maybe(..), Eq(..), Ord, Char, Monad, map
               , flip, undefined, (.), (&&), otherwise, ($), const, Bool(..)
               , (+), error, (++), fmap )

import qualified Data.Map as Map
import Data.Map ( Map )
import Data.Kind ( Type )
import Data.Set ( Set)
import qualified Data.Set as Set
import RandomType
import Test.QuickCheck ( (==>) )

import GHC.TypeLits ( Nat )

class Dummy (n :: Nat) where
  method :: ()

f1 >=> f3 = \x -> do y <- f1 x
                     f3 y

(>>>) = flip (.)

(|>) = flip ($)

empty = undefined
lookup = undefined
alter = undefined
union = undefined
mapEM = undefined
foldrEM = undefined

type Var = String

unionExprS = unionWithExprS const

data ListExprMap v
lookupLEM = undefined
lookupLM0 = undefined
lookupLM1 = undefined

{-# NOINLINE exf #-}
exf :: Int -> Int -> Char
exf = undefined
exf2 :: Int -> Char
exf2 = undefined

insertPM = undefined
matchPM = undefined
data PatMap v

lkMExprS0 = undefined

xtMExprS = undefined

instance Eq ExprS where
  VarS v1 == VarS v2 = v1 == v2
  AppS e1a e1b == AppS e2a e2b = e1a == e2a && e1b == e2b
  _ == _ = False

instance Eq Expr where
  (==) = go Map.empty Map.empty 0
    where
      go :: Map Var Int -> Map Var Int -> Int -> Expr -> Expr -> Bool
      go lhsmap rhsmap _ (Var v1) (Var v2)
        = case (Map.lookup v1 lhsmap, Map.lookup v2 rhsmap) of
            (Just level1, Just level2) -> level1 == level2
            (Nothing, Nothing)         -> v1 == v2
            _ matchPM False
      go lhsmap rhsmap next (App a b) (App c d) = go lhsmap rhsmap next a c && go lhsmap rhsmap next b d
      go lhsmap rhsmap next (Lam v1 e1) (Lam v2 e2) = go (Map.insert v1 next lhsmap) (Map.insert v2 next rhsmap) (next+1) e1 e2
      go _ _ _ _ _ = False

\end{code}

%format checktype(e) = "instance Dummy $( randomType ) where method = const () (" e ")"

%endif


\begin{document}

\special{papersize=8.5in,11in}
\setlength{\pdfpageheight}{\paperheight}
\setlength{\pdfpagewidth}{\paperwidth}

%\newcommand{\simon}[1]{[{\bf SLPJ}: {\color{red} #1}]}
%\newcommand{\js}[1]{{\bf JS}: {\color{olive} #1} {\bf End JS}}
%\newcommand{\rae}[1]{{\bf RAE}: {\color{dkblue} #1} {\bf End RAE}}
%\newcommand{\sg}[1]{{\bf SG}: {\color{darkbrown} #1} {\bf End SG}}
\newcommand{\simon}[1]{}
\newcommand{\js}[1]{}
\newcommand{\rae}[1]{}
\newcommand{\sg}[1]{}


\newcommand{\bv}[1]{\#_{#1}}    % Lambda-bound variable occurrence
\newcommand{\pv}[1]{\$_{#1}}    % Pattern variable binder
\newcommand{\pvo}[1]{\%_{#1}}   % Pattern variable occurrence

% Benchmark formatting hooks
\newcommand{\benchname}[1]{\texttt{#1}}
\newcommand{\insigdig}[1]{\ensuremath{\tilde{\text{#1}}}} % How to mark insignificant (within 2*σ) digits
\newcommand{\hackage}[1]{\varid{#1}\footnote{\url{https://hackage.haskell.org/package/#1}}}

%% Title information
\title%[Short Title]
{Triemaps that match}         %% [Short Title] is optional;
                                        %% when present, will be used in
                                        %% header instead of Full Title.
% \titlenote{with title note}             %% \titlenote is optional;
%                                         %% can be repeated if necessary;
%                                         %% contents suppressed with 'anonymous'
\subtitle{Technical Report}               %% \subtitle is optional
% \subtitlenote{with subtitle note}       %% \subtitlenote is optional;
%                                         %% can be repeated if necessary;
%                                         %% contents suppressed with 'anonymous'


%% Author information
%% Contents and number of authors suppressed with 'anonymous'.
%% Each author should be introduced by \author, followed by
%% \authornote (optional), \orcid (optional), \affiliation, and
%% \email.
%% An author may have multiple affiliations and/or emails; repeat the
%% appropriate command.
%% Many elements are not rendered, but should be provided for metadata
%% extraction tools.

%% Author with single affiliation.
\author{Simon Peyton Jones}
\affiliation{
  \institution{Epic Games}
  % \streetaddress{21 Station Rd.}
  \city{Cambridge}
  % \postcode{CB1 2FB}
  \country{UK}
}
\email{simon.peytonjones@@gmail.com}

\author{Sebastian Graf}
\affiliation{%
  \institution{Karlsruhe Institute of Technology}
  \city{Karlsruhe}
  \country{Germany}
}
\email{sebastian.graf@@kit.edu}

% Some conditional build stuff for handling the Appendix
% Both conditionals, main and appendix, will be set from the Makefile depending
% on the target (main, extended, appendix).
%if main

%% Abstract
%% Note: \begin{abstract}...\end{abstract} environment must come
%% before \maketitle command
\begin{abstract}
  The \emph{trie} data structure is a good choice for finite maps whose
  keys are data structures (trees) rather than atomic values. But what if we want
  the keys to be \emph{patterns}, each of which matches many lookup keys?
  Efficient matching of this kind is well studied in the theorem prover
  community, but much less so in the context of statically typed functional programming.
  Doing so yields an interesting new viewpoint --- and a practically useful design
  pattern, with good runtime performance.
\end{abstract}

%% \maketitle
%% Note: \maketitle command must come after title commands, author
%% commands, abstract environment, Computing Classification System
%% environment and commands, and keywords command.
\maketitle


\section{Introduction} \label{sec:intro}

Many functional languages provide \emph{finite maps} either as a
built-in data type, or as a mature, well-optimised library.  Generally the keys
of such a map will be small: an integer, a string, or perhaps a pair of integers.
But in some applications the key is large: an entire tree structure.  For example,
consider the Haskell expression
\begin{code}
  let x = a+b in ...(let y = a+b in x+y)....
\end{code}
We might hope that the compiler will recognise the repeated sub-expression
|(a+b)| and transform to
\begin{code}
  let x = a+b in ...(x+x)....
\end{code}
An easy way to do so is to build a finite map that maps the expression |(a+b)| to |x|.
Then, when encountering the inner |let|, we can look up the right hand side in the map,
and replace |y| by |x|.  All we need is a finite map keyed by syntax trees.

Traditional finite-map implementations tend to do badly in such applications, because
they are often based on balanced trees, and make the assumption that comparing two keys is
a fast, constant-time operation.  That assumption is false for large, tree-structured keys.

Another time that a compiler may want to look up a tree-structured key is
when rewriting expressions: it wants to see if any rewrite rule matches the
sub-expression in hand, and if so rewrite with the instantiated right-hand
side of the rule.
To do this we need a fast way to see if a target expression matches one of
the patterns in a set of (\varid{pattern},~\varid{rhs}) pairs.
If there is a large number of such
(\varid{pattern},~\varid{rhs}) entries to check, we would like to do so faster
than checking them one by one. Several parts of GHC, a Haskell compiler, need
matching lookup, and currently use an inefficient linear algorithm to do so.

In principle it is well known how to build a finite map for a deeply-structured
key: use a \emph{trie}.  The matching task is also well studied but, surprisingly,
only in the automated reasoning community (\Cref{sec:discrim-trees}): they use
so-called \emph{discrimination trees}.
In this paper we apply these
ideas in the context of a statically-typed functional programming language, Haskell.
This shift of context is surprisingly fruitful, and we make the following contributions:
\begin{itemize}
\item Following \citet{hinze:generalized}, we develop a standard pattern for
  a \emph{statically-typed triemap} for an arbitrary algebraic data type (\Cref{sec:basic}). In
  contrast, most of the literature describes untyped tries for a
  fixed, generic tree type.
  In particular:
  \begin{itemize}
    \item Supported by type classes, we can make good use of polymorphism to build triemaps
      for polymorphic data types, such as lists (\Cref{sec:class}).

    \item We cover the full range of operations expected for finite maps:
      not only |insert|ion and |lookup|, but |alter|, |union|, |foldr|, |map| and |filter|
      (\Cref{sec:basic}).

    \item We develop a generic optimisation for singleton maps that
      compresses leaf paths. Intriguingly, the resulting triemap
      \emph{transformer} can be easily mixed into arbitrary triemap definitions
      (\Cref{sec:singleton}).

   \end{itemize}
\item We show how to make our triemaps insensitive to \emph{$\alpha$-renamings} in
       keys that include binding forms (\Cref{sec:binders}).
       Accounting for $\alpha$-equivalence is not hard, but it is crucial for
       the applications in compilers.

\item We extend our triemaps to support \emph{matching} lookups
  (\Cref{sec:matching}).  This is an important step, because the only
  readily-available alternative is linear lookup. Our main
  contribution is to extend the established idea of tries keyed by
  arbitrary data types, so that it can handle matching too.

\item We present measurements that compare the performance of our triemaps (ignoring
  their matching capability) with traditional finite-map implementations in
  Haskell (\Cref{sec:eval}).
\end{itemize}
We discuss related work in \Cref{sec:related}.
Our contribution is not so much a clever new idea as an exposition of
some old ideas in a new context.  Nevertheless, we found it
surprisingly tricky to develop the ``right'' abstractions, such as
the |TrieMap| and |Matchable| classes, the singleton-and-empty map
data type, and the combinators we use in our instances.
These abstractions have been through \emph{many} iterations, and we
hope that by laying them out here we may shorten the path for others.

\section{The problem we address} \label{sec:problem}

Our general task is as follows: \emph{implement an efficient finite mapping
from keys to values, in which the key is a tree}.
Semantically, such a finite map is just a set of \emph{(key,value)}
pairs; we query the map by looking up a \emph{target}.
For example, the key might be a data type of syntax trees, defined like this:
%{
%if style == newcode
%format Expr = "Expr0"
%format App = "App0"
%format Lam = "Lam0"
%format Var = "Var0"
%endif
\begin{code}
type Var = String
data Expr = App Expr Expr | Lam  Var Expr | Var Var
\end{code}
Here the type |Var| represents variables; these can be compared for
equality and used as the key of a finite map.  Its definition is not important
for this paper, but for the sake of concreteness,
you may wish to imagine it is simply a string:
%}
% Convention: object language expressions like add x y in math mode
The data type |Expr| is capable of representing expressions like $(add\,x\,y)$ and
$(\lambda{}x.\,add\,x\,y)$. We will use this data type throughout the paper, because it
has all the features that occur in real expression data types: free variables like $add$,
represented by a |Var| node;
lambdas which can bind variables (|Lam|), and occurrences of those bound variables (|Var|);
and nodes with multiple children (|App|).  A real-world expression type would have
many more constructors, including literals, let-expressions and suchlike.

% A finite map keyed by such expressions is extremely useful.
% The Introduction gave the example of a simple common sub-expression
% elimination pass.
% GHC also does many lookups based on \emph{types} rather than
% \emph{expressions}.  For example, when implementing type-class
% instance lookup, or doing type-family reduction, GHC needs a map whose
% key is a type.  Both types and expressions are simply trees, and so are
% particular instances of the general task.

\subsection{Alpha-renaming} \label{sec:alpha-renaming}

In the context of a compiler, where the keys are expressions or types,
the keys may contain internal \emph{binders}, such as the binder $x$ in
$(\lambda{}x.x)$. If so, we would expect insertion and lookup to be insensitive
to $\alpha$-renaming, so we could, for example, insert with key $(\lambda{}x.x)$
and look up with key $(\lambda{}y.y)$, to find the inserted value.

\subsection{Lookup modulo matching} \label{sec:matching-intro}

Beyond just the basic finite maps we have described, our practical setting
in GHC demands more: we want to do a lookup that does \emph{matching}.  GHC supports
so-called \emph{rewrite rules}~\cite{rewrite-rules}, which the user can specify
in their source program, like this:
\begin{code}
prag_begin RULES "map/map"  forall f g xs.  map f (map g xs)
                                            =  map (f . g) xs prag_end
\end{code}
This rule asks the compiler to rewrite any target expression that matches the shape
of the left-hand side (LHS) of the rule into the right-hand side
(RHS).  We use the term \emph{pattern} to describe the LHS, and \emph{target} to describe
the expression we are looking up in the map.
The pattern is explicitly quantified over the \emph{pattern variables}
(here |f|, |g|, and |xs|) that
can be bound during the matching process.  In other words, we seek a substitution
for the pattern variables that makes the pattern equal to the target expression.
For example, if the program we are compiling contains the target expression
|map double (map square nums)|, we would like to produce a substitution
|[f ||-> double, g ||-> square, xs ||-> nums]| so that the substituted RHS
becomes |map (double . square) nums|; we would replace the former expression
with the latter in the code under consideration.

Of course, the pattern might itself have bound variables, and we would
like to be insensitive to $\alpha$-conversion for those. For example:
\begin{code}
prag_begin RULES "map/id"  map (\x -> x) = \y -> y prag_end
\end{code}
We want to find a successful match if we see a call |map (\y -> y)|,
even though the bound variable has a different name.

Now imagine that we have thousands of such rules.  Given a target
expression, we want to consult the rule database to see if any
rule matches.  One approach would be to look at the rules one at a
time, checking for a match, but that would be slow if there are many rules.
Similarly, GHC's lookup for
type-class instances and for type-family instances can have thousands
of candidates. We would like to find a matching candidate more efficiently
than by linear search.

\subsection{Non-solutions} \label{sec:ord}

At first sight, our task can be done easily: define a total order on |Expr|
and use a standard finite map library.
Indeed that works, but it is terribly slow.  A finite map is
implemented as a binary search tree; at every node of this tree,
we compare the key (an |Expr|, remember) with
the key stored at the node; if it is smaller, go left; if larger, go right. Each lookup
thus must perform a (logarithmic) number of
potentially-full-depth comparisons of two expressions.

Another possibility might be to hash the |Expr| and use the
hash-code as the lookup key.  That would make lookup much faster, but
it requires at least two full traversals of the key for every lookup:
one to compute its hash code for every lookup, and a full equality
comparison on a ``hit'' because hash-codes can collide.

But the killer is this: \emph{neither binary search trees nor hashing is compatible
with matching lookup}.  For our purposes they are non-starters.

What other standard solutions to matching lookup are there, apart from linear search?
The theorem proving and automated reasoning community has been working with huge sets
of rewrite rules, just as we describe, for many years.
They have developed term indexing techniques for the job \cite[Chapter 26]{handbook:2001},
which attack the same problem from a rather different angle, as we discuss in \Cref{sec:discrim-trees}.

\section{Tries} \label{sec:Expr}

A standard approach to a finite map in which the key has internal structure
is to use a \emph{trie}~\cite{Knuth1973}.
Generalising tries to handle an arbitrary algebraic data type as the key is
a well established, albeit under-used, idea \cite{connelly-morris,hinze:generalized}.
We review these ideas in this section.
Let us consider a simplified
form of expression:
\begin{code}
data Expr = Var Var | App Expr Expr
\end{code}
We omit lambdas for now,
so that all |Var| nodes represent free variables, which are treated as constants.
We will return to lambdas in \Cref{sec:binders}.

\subsection{The interface of a finite map} \label{sec:interface}

\begin{figure}
%{
%if style == poly
%format Map0 = Map
%format Dots = "\ldots"
%endif
\begin{code}
type TF v = Maybe v -> Maybe v

data Map0 k v = Dots  -- Keys k, values v
checktype(Map.empty      :: Map k v)
checktype(Map.insert     :: Ord k  => k -> v -> Map k v -> Map k v)
checktype(Map.lookup     :: Ord k  => k -> Map k v -> Maybe v)
checktype(Map.alter      ::  Ord k  => TF v -> k
                             -> Map k v -> Map k v)
checktype(Map.foldr      :: (v -> r -> r) -> r -> Map k v -> r)
checktype(Map.map        :: (v -> w) -> Map k v -> Map k w)
checktype(Map.unionWith  ::  Ord k  => (v->v->v)
                             -> Map k v -> Map k v -> Map k v)
checktype(Map.size       :: Map k v -> Int)
checktype(Map.compose    ::  Ord b  => Map b c -> Map a b -> Map a c)

infixr 1 >=>          -- Kleisli composition
(>=>) :: Monad m  => (a -> m b) -> (b -> m c)
                  -> a -> m c

infixr 1 >>>          -- Forward composition
(>>>)  :: (a -> b) -> (b -> c) -> a -> c

infixr 0 |>           -- Reverse function application
(|>)  :: a -> (a -> b) -> b
\end{code}
%}
\caption{API for various library functions leveraged in this work}
\label{fig:containers} \label{fig:library}
\end{figure}

Building on the design of widely
used functions in Haskell (see \cref{fig:containers}), we
seek these basic operations:
\begin{code}
emptyEM   :: ExprMap v
lookupEM  :: Expr -> ExprMap v -> Maybe v
atEM      :: Expr -> TF v -> ExprMap v -> ExprMap v
\end{code}
The lookup function |lookupEM|\footnote{We use short names |lookupEM| and |atEM|
  consistently in this paper to reflect the single-column format.
}
has a type that is familiar from every finite map.
The update function |atEM|, typically called |alter| in Haskell libraries,
changes the value stored at a particular key.
The caller provides a \emph{value transformation function} |TF v|, an
abbreviation for |Maybe v -> Maybe v| (see \Cref{fig:library}). This function
transforms the existing value associated with the key, if any (hence the input
|Maybe|), to a new value, if any (hence the output |Maybe|).
We can easily define |insertEM| and |deleteEM| from |atEM|:
\begin{code}
insertEM :: Expr -> v -> ExprMap v -> ExprMap v
insertEM e v = atEM e (\_ -> Just v)

deleteEM :: Expr -> ExprMap v -> ExprMap v
deleteEM e = atEM e (\_ -> Nothing)
\end{code}
You might wonder whether, for the purposes of this paper, we could just define |insertEM|,
leaving |alterEM| to the supplemental material of the extended version of this
paper~\cite{triemaps-extended}, but as we will see in \Cref{sec:alter}, our
approach using tries requires the generality of |alterEM|.

We also support other standard operations on finite maps,
with types analogous to those in \Cref{fig:library}, including |unionEM|, |mapEM|, and |foldrEM|.
%
% \begin{itemize}
% \item An efficient union operation to combine two finite maps into one:
% \begin{code}
% unionEM :: ExprMap v -> ExprMap v -> ExprMap v
% \end{code}
% \item A map operation to apply a function to the range of the finite map:
% \begin{code}
% mapEM :: (a -> b) -> ExprMap a -> ExprMap b
% \end{code}
% \item A fold operation to combine together the elements of the range:
% \begin{code}
% foldrEM :: (a -> b -> b) -> ExprMap a -> b -> b
% \end{code}
% \end{itemize}

\subsection{Tries: the basic idea} \label{sec:basic}

Here is a trie-based implementation for |Expr|:
%{
%if style == newcode
%format ExprMap = "ExprMap0"
%format EM = "EM0"
%format em_var = "em_var0"
%format em_app = "em_app0"
%format lookupEM = "lookupExpr0"
%format alterEM = "alterExpr0"
%format liftFT = "liftFT0"
%format emptyEM = "emptyExpr0"
%endif
\begin{code}
data ExprMap v
  = EM { em_var  :: Map Var v, em_app  :: ExprMap (ExprMap v) }
\end{code}
Here |Map Var v| is any standard finite map (e.g. in \hackage{containers})
keyed by |Var|, with values |v|.
One way to understand this slightly odd data type is to study its lookup function:
\begin{code}
lookupEM :: Expr -> ExprMap v -> Maybe v
lookupEM e (EM { em_var = var, em_app = app }) = case e of
  Var x      -> Map.lookup x var
  App e1 e2  ->  case lookupEM e1 app of
     Nothing  -> Nothing
     Just m1  -> lookupEM e2 m1
\end{code}
This function pattern-matches on the target |e|.  The |Var| alternative
says that to look up a variable occurrence, just look that variable up in the
|em_var| field.
But if the expression is an |App e1 e2| node, we first look up |e1|
in the |em_app| field, \emph{which returns an |ExprMap|}.  We then look up |e2|
in that map.  Each distinct |e1| yields a different |ExprMap| in which to look up |e2|.

We can substantially abbreviate this code, at the expense of making it more cryptic, thus:
%{
%if style == newcode
%format lookupEM = "lookupExpr2"
\begin{code}
lookupEM :: Expr -> ExprMap v -> Maybe v
  -- we need this type signature because the body is polymorphic recursive
\end{code}
%endif
\begin{code}
lookupEM (Var x)      = em_var  >>> Map.lookup x
lookupEM (App e1 e2)  = em_app  >>> lookupEM e1 >=> lookupEM e2
\end{code}
%}
The function |em_var :: ExprMap v -> Map Var v|
is the auto-generated selector that picks the |em_var| field from an |EM| record, and similarly |em_app|.
The functions |(>>>)| and |(>=>)| are right-associative forward composition
operators, respectively monadic and non-monadic,
that chain the individual operations together (see \Cref{fig:library}).
Finally, we have $\eta$-reduced the definition, by omitting the |m| parameter.
These abbreviations become quite worthwhile when we add more constructors, each with more fields,
to the key data type.

Notice that in contrast to the approach of \Cref{sec:ord}, \emph{we never compare two expressions
for equality or ordering}.  We simply walk down the |ExprMap| structure, guided
at each step by the next node in the target.
% (We typically use the term ``target'' for the
% key we are looking up in the finite map.)

This definition is extremely short and natural. But it embodies a hidden
complexity: \emph{it requires polymorphic recursion}. The recursive call to |lookupEM e1|
instantiates |v| to a different type than the parent function definition.
Haskell supports polymorphic recursion readily, provided you give a type signature to
|lookupEM|, but not all languages do.

\subsection{Modifying tries} \label{sec:alter} \label{sec:empty-infinite}

It is not enough to look up in a trie -- we need to \emph{build} them too.
First, we need an empty trie. Here is one way to define it:
%{
%if style == newcode
%format emptyEM = "emptyEM0"
%format foldrEM = "foldrEM0"
%format sizeEM = "sizeEM0"
%format ExprMap = "ExprMap0"
%format em_var = "em_var0"
%format em_app = "em_app0"
%format EM = "EM0"
%endif
\begin{code}
emptyEM :: ExprMap v
emptyEM = EM { em_var = Map.empty, em_app = emptyEM }
\end{code}
It is interesting to note that |emptyEM| is an infinite, recursive structure:
the |em_app| field refers back to |emptyEM|.  We will change this
definition in \Cref{sec:empty}, but it works perfectly well for now.
Next, we need to \emph{alter} a triemap \emph{at} some key, implemented as
|atEM|:
\footnote{The name |atEM| is borrowed from the |At| type class of the
\hackage{lens} package. Indeed, |ExprMap| could have an |At| instance if |atEM|
were generalised to an applicative traversal.}
\begin{code}
atEM :: Expr -> TF v -> ExprMap v -> ExprMap v
atEM e tf m@(EM { em_var = var, em_app = app }) = case e of
  Var x      -> m { em_var  = Map.alter tf x var }
  App e1 e2  -> m { em_app  = atEM e1 (liftTF (atEM e2 tf)) app }

liftTF :: (ExprMap v -> ExprMap v) -> TF (ExprMap v)
liftTF f Nothing    = Just (f emptyEM)
liftTF f (Just m)   = Just (f m)
\end{code}
%}
In the |Var| case, we must just update the map stored in the |em_var| field,
using the |Map.alter| function from \Cref{fig:containers}.
% in Haskell the notation ``|m { fld = e }|'' means the result
% of updating the |fld| field of record |m| with new value |e|.
In the |App| case we look up |e1| in |app|;
we should find a |ExprMap| there, which we want to alter with |tf|.
We can do that with a recursive call to |atEM|, using |liftTF|
for impedance-matching.

The |App| case shows why we need the generality of |atEM|.
Suppose we attempted to define an apparently-simpler |insertEM| operation.
Its equation for |(App e1 e2)| would look up |e1| --- and would then
need to alter that entry (an |ExprMap|, remember) with the result of
inserting |(e2,v)|.  So we are forced to define |atEM| anyway.

We can abbreviate the code for |atEM| using combinators, as we did in the case of
lookup, and doing so pays dividends when the key is a data type with
many constructors, each with many fields.  However, the details are
fiddly and not illuminating, so we omit them here.  Indeed, for the
same reason, in the rest of this paper we will typically omit the code
for |atEM|, though the full code is available in the
supplement~\cite{triemaps-extended}.

\subsection{Unions of maps}

A common operation on finite maps is to take their union:
\begin{code}
unionEM :: ExprMap v -> ExprMap v -> ExprMap v
\end{code}
In tree-based implementations of finite maps, such union operations can be tricky.
The two trees, which have been built independently, might not have the same
left-subtree/right-subtree structure, so some careful rebalancing may be required.
But for tries there are no such worries --
their structure is identical, and we can simply zip them together.  There is one
wrinkle: just as we had to generalise |insertEM| to |atEM|,
to accommodate the nested map in |em_app|, so we need to generalise |unionEM| to |unionWithEM|:
\begin{code}
unionWithEM  ::  (v -> v -> v)
             ->  ExprMap v -> ExprMap v -> ExprMap v
\end{code}
When a key appears on both maps, the combining function is used to
combine the two corresponding values.
With that generalisation, the code is as follows:
\begin{code}
unionWithEM f  (EM { em_var = var1, em_app = app1 })
               (EM { em_var = var2, em_app = app2 })
  = EM  { em_var = Map.unionWith f var1 var2
        , em_app = unionWithEM (unionWithEM f) app1 app2 }
\end{code}
It could hardly be simpler.

\subsection{Folds and the empty map} \label{sec:fold} \label{sec:empty}

The strange, infinite definition of |emptyEM| given in \Cref{sec:empty-infinite}
works fine (in a lazy language at least) for lookup, alteration, and union, but it fails
fundamentally when we want to \emph{iterate} over the elements of the trie.
For example, suppose we wanted to count the number of elements in the finite map; in |containers|
this is the function |Map.size| (\Cref{fig:containers}).  We might attempt:
%{
%if style == poly
%format undefined = "???"
%endif
\begin{code}
sizeEM :: ExprMap v -> Int
sizeEM (EM { em_var = var, em_app = app })
  = Map.size var {-" \; "-} + {-" \; "-} undefined
\end{code}
%}
We seem stuck because the size of the |app| map is not what we want: rather,
we want to add up the sizes of its \emph{elements}, and we don't have a way to do that yet.
The right thing to do is to generalise to a fold:
\begin{code}
foldrEM :: forall v. (v -> r -> r) -> r -> ExprMap v -> r
foldrEM k z (EM { em_var = var, em_app = app })
  = Map.foldr k z1 var
  where
    z1 = foldrEM kapp z (app :: ExprMap (ExprMap v))
    kapp m1 r = foldrEM k r m1
\end{code}
%}
In the binding for |z1| we fold over |app|, using
|kapp| to combine the map we find with the accumulator, by again
folding over the map with |foldrEM|.

But alas, |foldrEM| will never terminate!  It always invokes itself immediately
(in |z1|) on |app|; but that invocation will again recursively invoke
|foldrEM|; and so on forever.
The solution is simple: we just need an explicit representation of the empty map.
Here is one way to do it (we will see another in \Cref{sec:singleton}):
%{
%if style == newcode
%format ExprMap = "ExprMap1"
%format EmptyEM = "EmptyEM1"
%format EM = "EM1"
%format em_var = "em_var1"
%format em_app = "em_app1"
%endif
\begin{code}
data ExprMap v  = EmptyEM | EM { em_var :: ..., em_app :: ... }

emptyEM :: ExprMap v
emptyEM = EmptyEM

foldrEM :: (v -> r -> r) -> r -> ExprMap v -> r
foldrEM k z EmptyEM = z
foldrEM k z (EM { em_var = var, em_app = app })
  = Map.foldr k z1 var
  where
    z1 = foldrEM kapp z app
    kapp m1 r = foldrEM k r m1
\end{code}
Equipped with a fold, we can easily define the size function, and another
that returns the range of the map:
\begin{code}
sizeEM :: ExprMap v -> Int
sizeEM = foldrEM (\ _ n -> n+1) 0

elemsEM :: ExprMap v -> [v]
elemsEM = foldrEM (:) []
\end{code}
%}

\subsection{A type class for triemaps} \label{sec:generalised} \label{sec:class}

Since all our triemaps share a common interface, it is useful to define
a type class for them:
%{
%if style == poly
%format dots = "\ldots"
%else
%format dots = ""
%endif
\begin{code}
class Eq (TrieKey tm) => TrieMap tm where
   type TrieKey tm :: Type
   emptyTM      :: tm a
   lookupTM     :: TrieKey tm -> tm a -> Maybe a
   alterTM      :: TrieKey tm -> TF a -> tm a -> tm a
   foldrTM      :: (a -> b -> b) -> tm a -> b -> b
   unionWithTM  :: (a -> a -> a) -> tm a -> tm a -> tm a
   dots
\end{code}
%}
The class constraint |TrieMap tm| says that the type |tm| is a triemap, with operations
|emptyTM|, |lookupTM| etc.
The class has an \emph{associated type} \cite{associated-types}, |TrieKey tm|,
a type-level function that transforms the type of the triemap into
the type of \emph{keys} of that triemap.

Now we can witness the fact that |ExprMap| is a |TrieMap|, like this:
%{
%if style == poly
%format dots = "\ldots"
%else
%format dots = "foldrTM = undefined"
%endif
\begin{code}
instance TrieMap ExprMap where
  type TrieKey ExprMap = Expr
  emptyTM   = emptyEM
  lookupTM  = lookupEM
  alterTM   = atEM
  dots
\end{code}
%}
Having a class allows us to write helper functions that work for any triemap
implementation, such as
\begin{code}
insertTM :: TrieMap tm => TrieKey tm -> v -> tm v -> tm v
insertTM k v = alterTM k (\_ -> Just v)

deleteTM :: TrieMap tm => TrieKey tm -> tm v -> tm v
deleteTM k = alterTM k (\_ -> Nothing)
\end{code}
But that is not all.
Suppose our expressions had multi-argument apply nodes, |AppV|, thus
%{ So
%if style == poly
%format dots = "\ldots"
%else
%format dots = "Ctor"
%format Expr = "Expr1"
%endif
\begin{code}
data Expr = dots | AppV Expr [Expr]
\end{code}
%}
Then we would need to build a trie keyed by a \emph{list} of |Expr|.
A list is just another algebraic data type, built with nil and cons,
so we \emph{could} use exactly the same approach, thus
\begin{code}
lookupLEM :: [Expr] -> ListExprMap v -> Maybe v
\end{code}
But rather than define a |ListExprMap| for keys of type |[Expr]|,
and a |ListDeclMap| for keys of type |[Decl]|, etc, we would obviously prefer
to build a trie for lists of \emph{any type}, like this \cite{hinze:generalized}:
\begin{code}
instance TrieMap tm => TrieMap (ListMap tm) where
   type TrieKey (ListMap tm) = [TrieKey tm]
   emptyTM   = emptyLM
   lookupTM  = lookupLM
   ...

data ListMap tm v = LM  { lm_nil  :: Maybe v
                        , lm_cons :: tm (ListMap tm  v) }

emptyLM :: TrieMap tm => ListMap tm
emptyLM = LM { lm_nil = Nothing, lm_cons = emptyTM }

lookupLM :: TrieMap tm => [TrieKey tm] -> ListMap tm v -> Maybe v
lookupLM []      = lm_nil
lookupLM (k:ks)  = lm_cons >>> lookupTM k >=> lookupLM ks
\end{code}
The code for |alterLM| and |foldrLM| is routine. Notice that all of
these functions are polymorphic in |tm|, the triemap for the list elements.

\subsection{Singleton maps, and empty maps revisited} \label{sec:singleton}

Suppose we start with an empty map, and insert a value
with a key (an |Expr|) such as
\begin{spec}
  App (App (Var "f") (Var "x")) (Var "y")
\end{spec}
Looking at the code
for |atEM| in \Cref{sec:alter}, you can see that
because there is an |App| at the root, we will build an
|EM| record with an empty |em_var|, and an
|em_app| field that is... another |EM|
record.  Again the |em_var| field will contain an
empty map, while the |em_app| field is a further |EM| record.

In effect, the key is linearised into a chain of |EM| records.
This is great when there are a lot of keys with shared structure, but
once we are in a sub-tree that represents a \emph{single} key-value pair it is
a rather inefficient way to represent the key.  So a simple idea is this:
when an |ExprMap| represents a single key-value pair, represent it
directly as a key-value pair, like this:
\begin{code}
data ExprMap v  =  EmptyEM
                |  SingleEM Expr v   -- A single key/value pair
                |  EM { em_var :: ..., em_app :: ... }
\end{code}
But in the triemap for each new data type |X|,
we will have to tiresomely repeat these extra data constructors, |EmptyX| and |SingleX|.
For example we would have to add |EmptyList| and |SingleList| to the |ListMap| data type
of \Cref{sec:class}.
It is better instead to abstract over the enclosed triemap, as follows:%
\footnote{|SEMap| stands for \enquote{singleton or empty map}.}
\begin{code}
data SEMap tm v  =  EmptySEM
                 |  SingleSEM (TrieKey tm) v
                 |  MultiSEM  (tm v)

instance TrieMap tm => TrieMap (SEMap tm) where
  type TrieKey (SEMap tm) = TrieKey tm
  emptyTM   = EmptySEM
  lookupTM  = lookupSEM
  alterTM   = alterSEM
  ...
\end{code}
The code for lookup practically writes itself. We abstract over |Maybe|
with some |MonadPlus| combinators to enjoy forwards compatibility to
\Cref{sec:matching}:
\begin{code}
lookupSEM :: TrieMap tm => TrieKey tm -> SEMap tm v -> Maybe v
lookupSEM k m = case m of
  EmptySEM        -> mzero
  SingleSEM pk v  -> guard (k == pk) >> pure v
  MultiSEM m      -> lookupTM k m
\end{code}
Where |mzero| means |Nothing| and |pure| means |Just|. The |guard| expression
in the |SingleSEM| will return |Nothing| when the key expression |k| doesn't
equate to the pattern expression |pk|.
To test for said equality we require an |Eq (TrieKey tm)| instance, hence it is
a superclass of |TrieMap tm| in the |class| declaration in \Cref{sec:class}.

The code for alter is more interesting, because it governs the shift from
|EmptySEM| to |SingleSEM| and thence to |MultiSEM|:
\begin{code}
alterSEM  :: TrieMap tm
          => TrieKey tm -> TF v -> SEMap tm v -> SEMap tm v
alterSEM k tf EmptySEM = case tf Nothing of  Nothing  -> EmptySEM
                                             Just v   -> SingleSEM k v
alterSEM k1 tf (SingleSEM k2 v2) = if k1 == k2
  then case tf (Just v2) of
      Nothing  -> EmptySEM
      Just v'  -> SingleSEM k2 v'
  else case tf Nothing of
      Nothing  -> SingleSEM k2 v2
      Just v1  -> MultiSEM (insertTM k1 v1 (insertTM k2 v2 emptyTM))
alterSEM k tf (MultiSEM tm) = MultiSEM (alterTM k tf tm)
\end{code}
Adding a new item to a triemap can turn |EmptySEM| into |SingleSEM| and |SingleSEM|
into |MultiSEM|; and deleting an item from a |SingleSEM| turns it back into |EmptySEM|.
You might wonder whether we can shrink a |MultiSEM| back to a |SingleSEM| when it has
only one remaining element?
Yes we can, but it takes quite a bit of code, and it is far from clear
that it is worth doing so.

Finally, we need to re-define |ExprMap| and |ListMap| using |SEMap|:
\begin{code}
  type ExprMap       = SEMap ExprMap'
  data ExprMap' v    = EM { em_var  :: ..., em_app  :: ExprMap (ExprMap v) }

  type ListMap        = SEMap ListMap'
  data ListMap' tm v  = LM { lm_nil  :: ..., lm_cons :: tm (ListMap tm v) }
\end{code}
The auxiliary data types |ExprMap'| and |ListMap'| have only a single constructor, because
the empty and singleton cases are dealt with by |SEMap|.  We reserve the original,
un-primed, names for the user-visible |ExprMap| and |ListMap| constructors.

\subsection{Generic programming}\label{sec:generic}

We have not described a triemap \emph{library}; rather we have described a \emph{design pattern}.
More precisely, given a new algebraic data type |X|, we have described a systematic way
of defining a triemap, |XMap|, keyed by values of type |X|.
Such a triemap is represented by a record:
\begin{itemize}
\item Each \emph{constructor} |K| of |X| becomes a \emph{field} |x_k| in |XMap|.
\item Each \emph{field} of a constructor |K| becomes a \emph{nested triemap} in the type of the field |x_k|.
\item If |X| is polymorphic then |XMap| becomes a triemap transformer, like
  |ListMap| above.
\end{itemize}
Actually writing out all this boilerplate code is tiresome, and it can of course be automated.
One way to do so would be to
use generic or polytypic programming, and Hinze describes precisely this \cite{hinze:generalized}.
Another approach would be to use Template Haskell.

We do not develop either of these approaches here, because our focus is only the
functionality and expressiveness of the triemaps.  However, everything we do is compatible
with an automated approach to generating boilerplate code.

\section{Keys with binders} \label{sec:binders}
\begin{figure}
\begin{code}
type DBNum = Int
data DeBruijnEnv = DBE { dbe_next :: DBNum, dbe_env :: Map Var DBNum }

emptyDBE :: DeBruijnEnv
emptyDBE = DBE { dbe_next = 1, dbe_env = Map.empty }

extendDBE :: Var -> DeBruijnEnv -> DeBruijnEnv
extendDBE v (DBE { dbe_next = n, dbe_env = dbe })
  = DBE { dbe_next = n+1, dbe_env = Map.insert v n dbe }

lookupDBE :: Var -> DeBruijnEnv -> Maybe DBNum
lookupDBE v (DBE {dbe_env = dbe }) = Map.lookup v dbe
\end{code}
\caption{De Bruijn leveling}
\label{fig:debruijn}
\end{figure}

If our keys are expressions (in a compiler, say) they may contain binders,
and we want insert and lookup to be insensitive to $\alpha$-renaming.
That is the challenge we address next. Here is our data type |Expr| from
\Cref{sec:alpha-renaming}, which brings back binding semantics through the |Lam|
constructor:
\begin{code}
data Expr = App Expr Expr | Lam Var Expr | Var Var
\end{code}
The key idea is simple: we perform de Bruijn numbering on the fly,
renaming each binder to a natural number, from outside in.
So, when inserting or looking up a key $(\lambda x.\, foo~ (\lambda y.\, x+y))$ we
behave as if the key was $(\lambda.\, foo ~(\lambda. \bv{1} + \bv{2}))$, where
each $\bv{i}$ stands for an occurrence of the variable bound by the $i$'th
lambda, counting from the root of the expression. In effect, then, we behave as
if the data type was like this:
\begin{spec}
data Expr' = App Expr Expr | Lam Expr | FVar Var | BVar BoundKey
\end{spec}
Notice (a) the |Lam| node no longer has a binder and (b) there are
two sorts of |Var| nodes, one for free variables and one for bound
variables, carrying a |BoundKey| (see below). We will not actually
build a value of type |Expr'| and look that up in a trie keyed by |Expr'|;
rather, we are going to \emph{behave as if we did}. Here is the code
(which uses \Cref{fig:debruijn}):
\begin{code}
data ModAlpha a = A DeBruijnEnv a
type AlphaExpr = ModAlpha Expr
instance Eq AlphaExpr where ...

type BoundKey  = DBNum
type ExprMap = SEMap ExprMap'
data ExprMap' v
  = EM  {  em_fvar  :: Map Var v         -- Free vars
        ,  em_bvar  :: Map BoundKey v    -- Lambda-bound vars
        ,  em_app   :: ExprMap (ExprMap v)
        ,  em_lam   :: ExprMap v }

instance TrieMap ExprMap' where
  type TrieKey ExprMap' = AlphaExpr
  lookupTM = lookupEM
  ...

lookupEM :: AlphaExpr -> ExprMap' v -> Maybe v
lookupEM (A bve e) = case e of
  Var v -> case lookupDBE v bve of
    Nothing  -> em_fvar  >>> Map.lookup v
    Just bv  -> em_bvar  >>> Map.lookup bv
  App e1 e2  -> em_app   >>> lookupTM (A bve e1) >=> lookupTM (A bve e2)
  Lam v e    -> em_lam   >>> lookupTM (A (extendDBE v bve) e)

lookupClosedExpr :: Expr -> ExprMap v -> Maybe v
lookupClosedExpr e = lookupEM (A emptyDBE e)
\end{code}
We maintain a |DeBruijnEnv| (cf.~\cref{fig:debruijn}) that
maps each lambda-bound variable to its de Bruijn level \cite{debruijn},%
\footnote{
  The de Bruijn \emph{index} of the occurrence of a variable $v$ counts the number
  of lambdas between the occurrence of $v$ and its binding site.  The de Bruijn \emph{level}
  of $v$ counts the number of lambdas between the root of the expression and $v$'s binding site.
  It is convenient for us to use \emph{levels}.}
its |BoundKey|.
The expression we look up --- the first argument of |lookupEM| --- becomes an
|AlphaExpr|, which is a pair of a |DeBruijnEnv| and an |Expr|.
At a |Lam|
node we extend the |DeBruijnEnv|. At a |Var| node we
look up the variable in the |DeBruijnEnv| to decide whether it is
lambda-bound or free, and behave appropriately.%
\footnote{The implementation in the supplement~\cite{triemaps-extended} uses
more efficient |IntMap|s for mapping |BoundKey|. |IntMap| is itself a trie
data structure, so it could have made a nice \enquote{Tries all the way down}
argument. But we found it distracting to present here, hence regular ordered
|Map|.}

The construction of \Cref{sec:singleton}, to handle empty and singleton maps,
applies without difficulty to this generalised map. To use it
we must define an instance |Eq AlphaExpr| to satisfy the |Eq| super class constraint
on the trie key, so that we can instantiate |TrieMap ExprMap'|.
That |Eq AlphaExpr| instance simply equates two
$\alpha$-equivalent expressions in the standard way.
The code for |at| and |foldr| holds no new surprises either.

And that is really all there is to it: it is remarkably easy to extend the basic
trie idea to be insensitive to $\alpha$-conversion and even mix in trie
transformers such as |SEMap| at no cost other than writing two instance
declarations.

\section{Tries that match} \label{sec:matching}

A key advantage of tries over hash-maps and balanced trees is
that we can support \emph{matching} (\Cref{sec:matching-intro}).

\subsection{What ``matching'' means} \label{sec:matching-spec}

% Our overall goal is to build a \emph{matching trie} into which we can:
% \begin{itemize}
% \item \emph{Insert} a (pattern, value) pair; here the insertion key is a pattern.
% \item \emph{Look up} a target expression, and return all the values whose pattern \emph{matches} that expression.
% \end{itemize}
Semantically, a matching trie can be thought of as a set of \emph{entries},
each of which is a (pattern, value) pair.
What is a pattern? It is a pair $(vs,p)$ where
\begin{itemize}
\item $vs$ is a set of \emph{pattern variables}, such as $[a,b,c]$.
\item $p$ is a \emph{pattern expression}, such as $f\, a\, (g\, b\, c)$.
\end{itemize}
A pattern may of course contain free variables (not bound by the pattern), such as $f$ and $g$
in the above example, which are regarded as constants by the algorithm.
A pattern $(vs, p)$ \emph{matches} a target expression $e$ iff there is a unique substitution
$S$ whose domain is $vs$, such that $S(p) = e$.

We allow the same variable to occur more than once in the pattern.
For example, the pattern $([x], f~ x~ x)$
should match targets like $(f~ 1~ 1)$ or $(f ~(g~ v)~ (g ~v))$,
but not $(f~ 1~ (g~ v))$.  This ability is important if we are to use matching tries
to implement class or type-family lookup in GHC.

%
%For example, suppose we wanted to encode the rewrite rule
%%{
%%if style == newcode
%%format f = "exf"
%%format f2 = "exf2"
%%endif
%\begin{code}
%prag_begin RULES "foo" forall x. f x x = f2 x prag_end
%\end{code}
%%}
%We would perhaps associate the (pattern, value) pair
%$(([x], f~ x~ x), (|"foo"|, f2~ x))$ with this rule, so that the value
%$(|"foo"|, f2~ x)$ returned from a successful match of the pattern allows us to
%map back to the rule name and its right-hand side, for example.
%Here the pattern $([x], f~ x~ x)$ has a repeated variable $x$,
%and should match targets like $(f~ 1~ 1)$ or $(f ~(g~ v)~ (g ~v))$,
%but not $(f~ 1~ (g~ v))$.  This ability is important if we are to use matching tries
%to implement class or type-family look in GHC.

In implementation terms, we can characterise matching by the following type
class:
\begin{code}
class (Eq (Pat k), MonadPlus (Match k)) => Matchable k where
  type Pat    k :: Type
  type Match  k :: Type -> Type
  match :: Pat k -> k -> Match k ()
\end{code}
For any key type |k|, the |match| function takes a pattern of type |Pat k|,
and a key of type |k|, and returns a monadic match of type |Match k ()|, where
|Pat| and |Match| are associated types of |k|.
Matching can fail or can return many results, so |Match k| is a |MonadPlus|:
\begin{code}
  mzero :: MonadPlus m => m a
  mplus :: MonadPlus m => m a -> m a -> m a
\end{code}
To make this more concrete, here is a possible |Matchable| instance for |AlphaExpr|:
\begin{code}
instance Matchable AlphaExpr where
  type Pat    AlphaExpr = PatExpr
  type Match  AlphaExpr = MatchExpr
  match = matchE
\end{code}
Let's look at the pieces, one at a time.

\subsubsection{Patterns} \label{sec:patterns}

A pattern |PatExpr| over |AlphaExpr| can be defined like this:
\begin{code}
data PatExpr = P PatKeys AlphaExpr
type PatKeys = Map PatVar PatKey
type PatVar  = Var
type PatKey  = DBNum
\end{code}
A pattern |PatExpr| is a pair of an |AlphaExpr| and a |PatKeys| that maps
each of the quantified pattern variables to a canonical de Bruijn |PatKey|.
Just as in \Cref{sec:binders}, |PatKeys| make the pattern insensitive to
the particular names, and order of quantification, of the pattern variables.
We canonicalise the quantified pattern variables before starting a lookup,
numbering pattern variables in the order they appear in the expression in a
left-to-right scan.
For example
$$
\begin{array}{r@@{\hspace{5mm}}l}
\text{Original pattern} & \text{Canonical |PatExpr|} \\
|([a,b], f a b a)|  &  |P [a =-> 1, b =-> 2] (f a b a)|\\
|([x,g], f (g x)|  &  |P [x =-> 2, g =-> 1] (f (g x))|
\end{array}
$$

\subsubsection{The matching monad} \label{sec:matching-monad}

There are many possible implementation of the |MatchExpr| monad, but here is one:
\begin{code}
type MatchExpr v = MR (StateT Subst [] v)
type Subst = Map PatKey Expr
\end{code}
The |MatchExpr| type is isomorphic to |Subst -> [(v,Subst)]|;
matching takes a substitution for pattern variables (more precisely, their
canonical |PatKey|s), and yields a possibly-empty list of values paired
with an extended substitution.  Notice that the substitution binds pattern keys
to |Expr|, not |AlphaExpr|, because the pattern variables cannot mention lambda-bound
variables within the target expression.

The formulation in terms of |StateT| endows us with just the right
|Monad| and |MonadPlus| instances, as well as favorable performance
because of early failure on contradicting |match|es and the ability to
share work done while matching a shared prefix of multiple patterns.

The monad comes with some auxiliary functions that we will need later:
\begin{code}
runMatchExpr :: MatchExpr v -> [(Subst, v)]
liftMaybe    :: Maybe v -> MatchExpr v
refineMatch  :: (Subst -> Maybe Subst) -> MatchExpr ()
\end{code}
Their semantics should be apparent from their types.  For example, |runMatchExpr|
runs a |MatchExpr| computation, starting with an empty |Subst|, and
returning a list of all the successful |(Subst,v)| matches.

\subsubsection{Matching summary}

The implementation of |matchE| is entirely straightforward, using
simultaneous recursive descent over the pattern and target.
The code is given in the supplement~\cite{triemaps-extended}.

The key point is this: nothing in this section is concerned with
tries.  Here we are simply concerned with the mechanics of matching,
and its underlying monad.  There is ample room for flexibility. For
example, if the key terms had two kinds of variables (say type
variables and term variables) we could easily define |Match| to carry
two substitutions; or |Match| could return just the first result
rather than a list of all of them; and so on.

\subsection{The matching trie class} \label{sec:matching-trie-class}

The core of our matching trie is the class |MTrieMap|, which
generalises the |TrieMap| class of \Cref{sec:class}:
\begin{code}
class Matchable (MTrieKey tm) => MTrieMap tm where
  type MTrieKey tm  :: Type
  emptyMTM      :: tm v
  lookupPatMTM  :: MTrieKey tm -> tm v -> Match (MTrieKey tm) v
  alterPatMTM   :: Pat (MTrieKey tm) -> TF v -> tm v -> tm v
\end{code}
The lookup function takes a key of type |MTrieKey tm| as before, but
it returns something in the |Match| monad, rather than the |Maybe| monad.
The |alterPatMTM| takes a \emph{pattern} (rather than just a key), of type
|Pat (MTrieKey tm)|, and alters the trie's value at that pattern.%
\footnote{ Remember, a matching trie represents a set of (pattern,value) pairs.}

We can generalise |SEMap| (\Cref{sec:singleton}) in a similar way:
\begin{code}
data MSEMap tm v  =  EmptyMSEM
                  |  SingleMSEM (Pat (MTrieKey tm)) v
                  |  MultiMSEM  (tm v)

instance MTrieMap tm => MTrieMap (MSEMap tm) where
  type MTrieKey (MSEMap tm) = MTrieKey tm
  emptyMTM      = EmptyMSEM
  lookupPatMTM  = lookupPatMSEM
  alterPatMTM   = alterPatMSEM
\end{code}
Notice that |SingleMSEM| contains a \emph{pattern},
not merely a \emph{key}, unlike |SingleSEM| in \Cref{sec:singleton}.
The code for |lookupPatMSEM| and |alterPatMSEM| is straightforward;
we give the former here, leaving the latter for the supplement~\cite{triemaps-extended}
\begin{code}
lookupPatMSEM :: MTrieMap tm  => MTrieKey tm -> MSEMap tm a
                              -> Match (MTrieKey tm) a
lookupPatMSEM k  EmptyMSEM           = mzero
lookupPatMSEM k  (MultiMSEM m)       = lookupPatMTM k m
lookupPatMSEM k  (SingleMSEM pat v)  = match pat k >> pure v
\end{code}
Notice the call to |mzero| to make the lookup fail if the map is empty; and, in the
|SingleMSEM| case, the call |match| to match the pattern against the key.

\subsection{Matching tries for |Expr|}

Next, we show how to implement a matching triemap for our running
example, |AlphaExpr|.
The data type follows closely the pattern we developed for |ExprMap| (\Cref{sec:binders}):
\begin{code}
type MExprMap = MSEMap MExprMap'

data MExprMap' v
  = MM  {  mm_fvar  :: Map Var v        -- Free var
        ,  mm_bvar  :: Map BoundKey v   -- Bound var
        ,  mm_pvar  :: Map PatKey v     -- Pattern var
        ,  mm_app   :: MExprMap (MExprMap v)
        ,  mm_lam   :: MExprMap v }

instance MTrieMap MExprMap' where
  type MTrieKey MExprMap' = AlphaExpr
  emptyMTM      = ... -- boring
  lookupPatMTM  = lookupPatMM
  alterPatMTM   = alterPatMM
\end{code}
The main difference is that we add an extra field |mm_pvar| to |MExprMap'|,
for occurrences of a pattern variable.  You can see how this field is used
in the lookup code:
\begin{code}
lookupPatMM :: forall v. AlphaExpr -> MExprMap' v -> MatchExpr v
lookupPatMM ae@(A bve e) (MM { .. })
  = rigid `mplus` flexi
  where
    rigid = case e of
      Var x      -> case lookupDBE x bve of
        Just bv  -> mm_bvar  |>   liftMaybe . Map.lookup bv
        Nothing  -> mm_fvar  |>   liftMaybe . Map.lookup x
      App e1 e2  -> mm_app   |>   lookupPatMTM (A bve e1)
                             >=>  lookupPatMTM (A bve e2)
      Lam x e    -> mm_lam   |>   lookupPatMTM (A (extendDBE x bve) e)

    flexi = mm_pvar |> IntMap.toList |> map match_one |> msum

    match_one :: (PatVar,v) -> MatchExpr v
    match_one (pv, v) = matchPatVarE pv ae >> pure v
\end{code}
Matching lookup on a trie matches the target
expression against \emph{all patterns the trie represents}.
The |rigid| case is no different from exact lookup; compare the
code for |lookupEM| in \Cref{sec:binders}.  The only difference is that we need
|liftMaybe| (from \Cref{sec:matching-monad}) to
take the |Maybe| returned by |Map.lookup| and lift it into the |MatchExpr| monad.

The |flexi| case handles the triemap entries whose pattern is simply one of
the quantified pattern variables; these entries are stored in the new |mm_pvar| field.
We enumerate these entries with |toList|, to get a list of |(PatVar,v)| pairs,
match each such pair against the target with |match_one|, and finally accumulate
all the results with |msum|.  In turn |match_one| uses |matchParVarE| to match
the pattern variable with the target and, if successful, returns corresponding
value |v|.

The |matchPatVarE| function does the heavy lifting, using some
simple auxiliary functions whose types are given below:
\begin{code}
matchPatVarE :: PatKey -> AlphaExpr -> MatchExpr ()
matchPatVarE pv (A bve e) = refineMatch $ \ms ->
  case Map.lookup pv ms of
    Nothing   -- pv is not bound
      |  noCaptured bve e  -> Just (Map.insert pv e ms)
      |  otherwise         -> Nothing
    Just sol  -- pv is already bound
      |  noCaptured bve e
      ,  eqExpr e sol      -> Just ms
      |  otherwise         -> Nothing

eqExpr      :: Expr -> Expr -> Bool
noCaptured  :: DeBruijnEnv -> Expr -> Bool
\end{code}
To match a pattern variable |pv| against an expression |(A bve e)|,
we first look up |pv| in the current substitution (obtained from the
|MatchExpr| state monad).  If |pv| is not bound we simply extend the
substitution.

But wait!  Consider matching the pattern |([p], \x -> p)|
against the target |(\y -> 3)|.  That's fine: we should succeed, binding
|p| to |3|.  But suppose we match that same pattern against target |(\y -> y)|.
It would be nonsense to ``succeed'' binding |p| to |y|, because |y| is
locally bound within the target.  Hence the |noCaptured| test, which
returns |True| iff the expression does not mention any of the locally-bound
variables.

If |pv| is already bound in the substitution, we have a repeated pattern
variable (see \Cref{sec:matching-spec}), and we must check that
the target expression is equal (using |eqExpr|) to the one already bound to |pv|.
Once again, however, we must check that the target does not contain any locally-bound
variables, hence the |noCaptured| check.

|lookupPatMM| is the trickiest case.  The code for |alterPatMM|, and
the other operations of the class, is very straightforward, and is given
in the supplement~\cite{triemaps-extended}.

\subsection{The external API} \label{sec:match-api}

The matching tries we have described so far use canonical pattern keys,
a matching monad, and other machinery that should be hidden from the client.
We seek an external API more like this:
\begin{code}
type PatMap :: Type -> Type
alterPM   :: ([Var], Expr) -> TF v -> PatMap v -> PatMap v
lookupPM  :: Expr -> PatMap v -> [(PatSubst, v)]
type PatSubst = [(Var,Expr)]
\end{code}
When altering a |PatMap|, we supply a client-side pattern, which is
just a pair |([Var],Expr)| of the quantified pattern variables and the pattern.
When looking up in a |PatMap|, we supply a target expression, and get back
a list of matches, each of which is a pair of the value and the substitution
for those original pattern variables that made the pattern equal to the target.

So |alterPM| must canonicalise the client-side pattern variables
before altering the trie; that is easy enough.
But how can |lookupPM| recover the client-side |PatSubst|?
Somehow we must remember the canonicalisation used when \emph{inserting}
so that we can invert it when \emph{matching}.   For example, suppose we insert the two
(pattern, value pairs)
$$
|(([p], f p True), v1)| \quad \text{and} \quad |(([q], f q False), v2)|
$$
Both patterns will canonicalise their (sole) pattern variable to the De Bruin index 1.
So if we look up the target |(f e True)| the |MatchExpr| monad will produce a
final |Subst| that maps |[1 =-> e]|, paired with the value |v1|.  But we want to return
|([("p",e)], v1)| to the client, a |PatSubst| that uses the client variable |"p"|, not
the internal index 1.

The solution is simple enough: \emph{we store the mapping in the triemap's domain},
along with the values, thus:
\begin{code}
type PatMap v = MExprMap (PatKeys, v)
\end{code}
Now the code writes itself. Here is |alterPM|:
\begin{code}
alterPM :: forall v. ([Var],Expr) -> TF v -> PatMap v -> PatMap v
alterPM (pvars, e) tf pm = alterPatMTM pat ptf pm
  where
    pks :: PatKeys = canonPatKeys pvars e

    pat :: PatExpr = P pks (A emptyDBE e)

    ptf :: TF (PatKeys, v)
    ptf Nothing       = fmap (\v -> (pks,v)) (tf Nothing)
    ptf (Just (_,v))  = fmap (\v -> (pks,v)) (tf (Just v))

canonPatKeys :: [Var] -> Expr -> PatKeys
\end{code}
The auxiliary function |canonPatKeys| takes the client-side pattern |(pvars,e)|,
and returns a |PatKeys| (\Cref{sec:patterns}) that maps each pattern variable
to its canonical de Bruijn index. |canonPatKeys| is entirely straightforward:
it simply walks the expression, numbering off the pattern variables in
left-to-right order.

Then we can simply call the internal |alterPatMTM| function,
passing it a canonical |pat :: PatExpr| and
a transformer |ptf :: TF (PatKeys,v)| that will pair the |PatKeys| with
the value supplied by the user via |tf :: TF v|.
Lookup is equally easy:
\begin{code}
lookupPM  :: Expr -> PatMap v -> [(PatSubst, v)]
lookupPM e pm
  = [ (Map.toList (subst `Map.compose` pks), x)
    | (subst, (pks, x)) <-  runMatchExpr $
                            lookupPatMTM (A emptyDBE e) pm ]
\end{code}
We use |runMatchExpr| to get a list of successful matches, and then pre-compose
(see \Cref{fig:containers}) the internal |Subst| with the |PatKeys| mapping that
is part of the match result. We turn that into a list to get the client-side
|PatSubst|. The only tricky point is what to do with pattern variables that are
not substituted. For example, suppose we insert the pattern |([p,q], f p)|. No
lookup will bind |q|, because |q| simply does not appear in the pattern. One
could reject this on insertion, but here we simply return a |PatSubst| with no
binding for |q|.

\subsection{Most specific match and unification} \label{sec:most-specific}

It is tempting to ask: can we build a lookup that returns only the \emph{most
specific} matches? And can we build a lookup that returns all values whose
patterns \emph{unify} with the target. Both would have useful applications, in
GHC at least.

However, both seem difficult to achieve. All our attempts became mired in
complexity, and we leave this for further work, and as a challenge for the
reader. We outline some of the difficulties of unifying lookup in Appendix B of
the extended version of this paper~\cite{triemaps-extended}.

\section{Evaluation} \label{sec:eval}

% For prototyping and getting a feel of the schema:
%\pgfplotstableread[row sep=\\,col sep=&]{
%  name                    & id & ExprMap & Map  & HashMap \\
%  \benchname{lookup}      & 1  & 1.00    & 1.06 & 1.64    \\
%  \benchname{lookup\_lam} & 2  & 1.00    & 7.82 & 1.95    \\
%  \benchname{fromList}    & 3  & 1.00    & 1.00 & 2.26    \\
%  \benchname{union}       & 4  & 1.00    & 1.09 & 1.06    \\
%}\benchdata
\pgfplotstableread{bench-plot.txt}\benchdata

\begin{figure}[h]
\begin{tikzpicture}
\begin{axis}[
  ybar,
  bar width=7pt,
  height=6.5cm,
  width=8cm,
  %
  % Set up y axis
  ymin=0,
  ymax=2,
  ylabel={Relative time (lower is better)},
  % Highlight baseline https://tex.stackexchange.com/a/133760/52414
  ytick={0.5,1.5,2.0},
  extra y ticks=1, %
  extra y tick style={grid=major, grid style={solid,black}},
  yticklabel={\pgfmathprintnumber{\tick}x},
  %
  % Set up x axis
  xmin=0.5,
  xmax=4.5,
  xtick=data,
  xticklabels from table={\benchdata}{name},
  x tick label style={font=\small},
  major x tick style={opacity=0},    % hide x ticks
  x tick label style={yshift=0.5em}, % and use the space for labels
  %
  % Indicate clipped bars by \cdots
  visualization depends on=rawy \as \rawy,
  nodes near coords={%
    \pgfmathfloatparsenumber{\rawy}%
    \let\rawy\pgfmathresult%
    \pgfmathfloatparsenumber{\pgfkeysvalueof{/pgfplots/ymax}}%
    \pgfmathfloatlessthan{\pgfmathresult}{\rawy}%
    \ifpgfmathfloatcomparison%
      $\cdots$%
    \fi%
    \pgfmathprintnumber[precision=2,fixed zerofill]{\rawy}
  },
  restrict y to domain*={
    \pgfkeysvalueof{/pgfplots/ymin}:\pgfkeysvalueof{/pgfplots/ymax}
  },
  % Make node labels smaller
  every node near coord/.append style={
    font=\small,
    anchor=west,
    rotate=90,
  },
]

\addplot table[x=id,y=ExprMap]{\benchdata};
\addplot table[x=id,y=Map]{\benchdata};
\addplot table[x=id,y=HashMap]{\benchdata};

\legend{TM,OM,HM}
\end{axis}
\end{tikzpicture}
\caption{Benchmarks comparing our trie map (TM)
  to ordered maps (OM) and hash maps (HM)}
\label{fig:plot}
\end{figure}

So far, we have seen that trie maps offer a significant advantage over other
kinds of maps like ordered maps or hash maps: the ability to do a matching
lookup (in \Cref{sec:matching}). In this section, we will see that query
performance is another advantage. Our implementation of trie maps in Haskell
can generally compete in performance with other map data structures, while
significantly outperforming traditional map implementations on some operations.
Not bad for a data structure that we can also extend to support matching lookup!

We took runtime measurements of the (non-matching) |ExprMap| data
structure on a selection of workloads, conducted using the \hackage{criterion}
benchmarking library.%
\footnote{The benchmark machine runs Ubuntu 18.04 on an Intel Core i5-8500 with
16GB RAM. All programs were compiled on GHC 9.0.2 with \texttt{-O2
-fproc-alignment=64} to eliminate code layout flukes and run with \texttt{+RTS
-A128M -RTS} for 128MB space in generation 0 in order to prevent major GCs from
skewing the results.}
\Cref{fig:plot} presents a quick overview of the results.

Appendix A of the extended version of this paper~\cite{triemaps-extended}
featurs a more in-depth analysis and finer runtime as well as space measurements
and indicators for statistical significance.

\subsubsection*{Setup}
All benchmarks except \benchname{fromList} are handed a pre-built
map containing 10000 expressions, each consisting of roughly 100 |Expr| data
constructors drawn from a pseudo-random source with a fixed (and thus
deterministic) seed.

We compare three different non-matching map implementations, simply because we
were not aware of other map data structures with matching lookup modulo
$\alpha$-equivalence and we wanted to compare apples to apples.
The |ExprMap| forms the baseline. Asymptotics are given with respect to map
size $n$ and key expression size $k$:

\begin{itemize}
  \item |ExprMap| (designated ``TM'' in \Cref{fig:plot}) is the trie map
        implementation from this paper. Insertion and lookup perform at most
        one full traversal of the key, so performance should scale with
        $\mathcal{O}(k)$.
  \item |Map Expr| (designated ``OM'') is the ordered map implementation from
        the mature, well-optimised \hackage{containers} library. It uses size
        balanced trees under the hood \cite{adams}. Thus, lookup and insert
        operations incur an additional log factor in the map size $n$, for a
        total of $\mathcal{O}(k \log n)$ factor compared to both other maps.
  \item |HashMap Expr| (designated ``HM'') is an implementation of hash array
        mapped tries \cite{hamt} from the \hackage{unordered-containers}
        library. Like |ExprMap|, map access incurs a full traversal of the key
        to compute a hash and then a $\mathcal{O}(\log_{32} n)$ lookup in the
        array mapped trie, as well as an expected constant number of key
        comparisons to resolve collisions. Consequently, lookup and insert
        are both in $\mathcal{O}(k + \log_{32} n)$, where the log summand can
        effectively be treated like a constant for all intents and purposes.
\end{itemize}

Some clarification as to what our benchmarks measure:

\begin{itemize}
  \item The \benchname{lookup} benchmark looks up every
        expression that is part of the map. So for a map of size 10000, we
        perform 10000 lookups of expressions each of which have approximately size
        100.
  \item \benchname{lookup\_lam} is like \benchname{lookup}, but wraps a shared
        prefix of 100 layers of |(Lam "$")| around each expression.
  \item \benchname{fromList} benchmarks a naïve |fromList|
        implementation on |ExprMap| against the tuned |fromList| implementations
        of the other maps, measuring map creation performance from batches.
\end{itemize}

\subsubsection*{Querying}
The results suggest that |ExprMap| is about as fast as |Map Expr| for
completely random expressions in \benchname{lookup}. But in a more realistic
scenario, at least some expressions share a common prefix, which is what
\benchname{lookup\_lam} embodies. There we can see that |ExprMap| wins against
|Map Expr| by a huge margin: |ExprMap| looks at the shared prefix exactly
once on lookup, while |Map| has to traverse the shared prefix of length
$\mathcal{O}(k)$ on each of its $\mathcal{O}(\log n)$ comparisons.

Although |HashMap| loses on most benchmarks compared to |ExprMap| and |Map|, most
measurements were consistently at most a factor of two slower than |ExprMap|.
We believe that is due to the fact that it is enough to traverse the
|Expr| twice during lookup barring any collisions (hash and then equate with the
match), thus it is expected to scale similarly as |ExprMap|. Thus, both |ExprMap|
and |HashMap| perform much more consistently than |Map|.

\subsubsection*{Modification}
While |ExprMap| consistently wins in query performance, the edge is melting into
insignificance for \benchname{fromList} and \benchname{union}. One reason is
the uniform distribution of expressions in these benchmarks, which favors |Map|.
Still, it is a surprise that the naïve |fromList| implementations of |ExprMap| and
|Map| as list folds beat the one of |HashMap|, although the latter has a tricky,
performance-optimised implementation using transient mutability.

What would a non-naïve version of |fromList| for |ExprMap| look like? Perhaps
the process could be sped up considerably by partitioning the input list
according to the different fields of |ExprMap| and then calling
the |fromList| implementations of the individual fields in turn. The process
would be very similar to discrimination sort~\cite{discr-sort}, which is a
generalisation of radix sort to tree-like data and very close to tries.
Indeed, the \hackage{discrimination} library provides such an optimised
$\mathcal{O}(n)$ |toMap| implementation for |Map|.

\section{Related work} \label{sec:related}

\subsection{Matching triemaps in automated reasoning} \label{sec:discrim-trees}

Matching triemaps have been used in the automated reasoning community for
decades, where they are recognised as a kind of \emph{term
index}~\cite{handbook:2001}. An automated reasoning system has
hundreds or thousands of axioms, each of which is quantified over
some variables (just like the RULEs described in \Cref{sec:matching-intro}). Each of these
axioms might apply at any sub-tree of the term under consideration, so efficient
matching of many axioms is absolutely central to the performance of these systems.

This led to a great deal of work on so-called \emph{discrimination trees}, starting
in the late 1980's, which is beautifully surveyed in the Handbook of Automated Reasoning
\cite[Chapter 26]{handbook:2001}.
All of this work typically assumes a single, fixed, data type of ``first order terms''
like this\footnote{Binders in terms do not seem to be important
in these works, although they could be handled fairly easily by a de Bruijn pre-pass.}
\begin{code}
  data MTrieKey = Node Fun [MTrieKey]
\end{code}
where |Fun| is a function symbol, and each such function symbol has a fixed arity.
Discrimination trees are described by imagining
a pre-order traversal that (uniquely, since function symbols have fixed arity)
converts the |MTrieKey| to a list of type |[Fun]|, and treating that as the key.
The map is implemented like this:
\begin{code}
  data DTree v = DVal v | DNode (Map Fun DTree)

  lookupDT :: [Fun] -> DTree v -> Maybe v
  lookupDT []      (DVal v)   = Just v
  lookupDT (f:fs)  (DNode m)  = case Map.lookup f m of
    Just dt -> lookupDT fs dt
    Nothing -> Nothing
  lookupDT _       _          = Nothing
\end{code}
Each layer of the tree branches on the first |Fun|, and looks up
the rest of the |[Fun]| in the appropriate child.
Extending this basic setup with matching is done by some kind of backtracking.

Discrimination trees are heavily used by theorem provers, such as Coq, Isabelle, and Lean.
Moreover, discrimination trees have been further developed in a number of ways.
Vampire uses \emph{code trees} which are a compiled form of discrimination
tree that stores abstract machine instructions, rather than a data structure
at each node of the tree \cite{voronkov:vampire}.
Spass \cite{spass} uses \emph{substitution trees} \cite{substitution-trees},
a refinement of discrimination trees that can share common \emph{sub-trees}
not just common \emph{prefixes}. (It is not clear whether the extra complexity of
substitution trees pays its way.)  Z3 uses \emph{E-matching code trees}, which solve
for matching modulo an ever-growing equality relation, useful in saturation-based
theorem provers.  All of these techniques except E-matching are surveyed in
\citet{handbook:2001}.

If we applied our ideas to |MTrieKey| we would get a single-field triemap which
(just like |lookupDT|) would initially branch on |Fun|, and then go though
a chain of |ListMap| constructors (which correspond to the |DNode| above).
You have to squint pretty hard  --- for example, we do the pre-order traversal on the fly
--- but the net result is very similar, although
it is arrived at by entirely different thought process.
% \begin{itemize}
% \item We present our triemaps as a library written in a statically typed functional
%  language, whereas the discrimination tree literature tends to assume an implementation in C,
%   and gives algorithms in pseudocode.
%
% \item Our triemaps provide a full range of operations, including alter, union, and fold,
%  wheres the automated-reasoning applications focus almost exclusively on insert and lookup.
%
% \item We build triemaps for many
% different data types, whereas the discrimination tree literature tends to assume
% a single built-in data type of terms.
%
% \item We use type classes and polymorphism to make it easy to build triemaps
%   over polymorphic types like lists (\Cref{sec:class}).
% \end{itemize}

Many of the insights of the term indexing world re-appear, in different guise,
in our triemaps.   For example, when a variable is repeated in a pattern we can
eagerly check for equality during the match, or instead gather an equality constraint
and check those constraints at the end \cite[Section 26.14]{handbook:2001}.

A related application of matching tries appear in
\cite[Section 2.2]{linked-visualisations}, where \emph{eliminators} express
both parameter-binding and pattern-matching in a single Core language construct,
with a semantics not unlike GHC's own @-XLambdaCase@ extension.
They realise that their big-step interpreter implements eliminators via special
generalised tries that can express variable matching -- which corresponds to our
triemaps applied to linear patterns.

\subsection{Haskell triemaps}

Trie data structures have found their way into numerous Haskell packages over time.
There are trie data structures that are specific to |String|, like the
\hackage{StringMap} package, or polymorphically, requiring just a type class for
trie key extraction, like the \hackage{TrieMap} package. None of these
libraries describe how to index on expression data structures modulo
$\alpha$-equivalence or how to perform matching lookup.

Memoisation has been a prominent application of tries in Haskell
\cite{hinze:memo,conal:blog1,conal:blog2}.
Given a function |f|, the idea is to build an \emph{infinite},
lazily-evaluated trie, that maps every possible argument |x| to (a thunk for)
$|(f x)|$.  Now, a function call becomes a lookup in the trie.
These ideas are implemented in the \hackage{MemoTrie} library, and have been generalised
to representable functors in the \hackage{representable-tries} library.
For memo tries, operations like alter, insert, union, and fold are all
irrelevant: the infinite trie is built once, and then used only for lookup.

A second strand of work concerns data type generic, or polytypic, approaches to
generating tries, which nicely complements the design-pattern approach
of this paper (\Cref{sec:generic}).
\citet{hinze:generalized} describes the polytypic approach,
for possibly parameterised and nested data types in some detail, including the
realisation that we need |atEM| and |unionWithEM| in order to define |insertEM| and
|unionEM|.
The aforementioned \varid{MemoTrie}, \varid{representable-tries} and
\hackage{generic-trie} libraries generate trie implementations polytypically.
For our approach to do the same, we would need to find a good way to specify
binding structure (as in |Lam|) in a polytypic setting, which thus far has been
elusive.

The \hackage{twee-lib} library defines a simple term index data structure based
on discrimination trees for the \varid{twee} equation theorem prover. We would
arrive at a similar data structure in this paper had we started from an
expression data type
\begin{code}
data Expr = App Con [Expr] | Var Var
\end{code}
In contrast to our |ExprMap|, \varid{twee}'s |Index| does path compression not
only for paths ending in leaves (as we do) but also for internal paths, as is
common for radix trees.

It is however unclear how to extend \varid{twee}'s |Index| to support
$\alpha$-equivalence, hence we did not consider it for our benchmarks in
\Cref{sec:eval}.

\begin{acks}
We warmly thank Leonardo de Moura and Edward Yang for their very helpful feedback.
\end{acks}

% \subsection{Notes from Sebastian}
%
% \begin{itemize}
% \item Using a FSM; e.g \emph{Interpreted Pattern Match Execution} by Jeff Niu, a UW undergrad intern at Google.  https://docs.google.com/presentation/d/1e8MlXOBgO04kdoBoKTErvaPLY74vUaVoEMINm8NYDds/edit?usp=sharing
%
% \item Matching multiple strings.
% \end{itemize}

\section{Conclusion}

We presented trie maps as an efficient data structure for representing a set of
expressions modulo $\alpha$-equivalence, re-discovering polytypic deriving
mechanisms described by~\citet{hinze:generalized}. Subsequently, we showed how to
extend this data structure to make it aware of pattern variables in order to
interpret stored expressions as patterns. The key innovation is that the
resulting trie map allows efficient matching lookup of a target expression
against stored patterns. This pattern store is quite close to discrimination
trees~\cite{handbook:2001}, drawing a nice connection to term indexing problems
in the automated theorem proving community.

\bibliography{refs}

% Closing main part of the paper
%endif
% Now the appendix
%if appendix
\appendix
%include appendix.lhs
%endif

\end{document}
