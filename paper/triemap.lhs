% -*- latex -*-

% Links
% https://stackoverflow.com/questions/16084788/generic-trie-haskell-implementation
% Hinze paper: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.8.4069

%% For double-blind review submission, w/o CCS and ACM Reference (max submission space)
% \documentclass[acmsmall,review]{acmart}\settopmatter{printfolios=true,printccs=false,printacmref=false}
%% For double-blind review submission, w/ CCS and ACM Reference
%\documentclass[acmsmall,review,anonymous]{acmart}\settopmatter{printfolios=true}
%% For single-blind review submission, w/o CCS and ACM Reference (max submission space)
% \documentclass[acmsmall,review]{acmart}\settopmatter{printfolios=true,printccs=false,printacmref=false}
%% For single-blind review submission, w/ CCS and ACM Reference
%\documentclass[acmsmall,review]{acmart}\settopmatter{printfolios=true}
%% For final camera-ready submission, w/ required CCS and ACM Reference
\documentclass[acmsmall,dvipsnames]{acmart}

%% Journal information
%% Supplied to authors by publisher for camera-ready submission;
%% use defaults for review submission.
% \acmJournal{PACMPL}
% \acmVolume{1}
% \acmNumber{ICFP} % CONF = POPL or ICFP or OOPSLA
% \acmArticle{1}
% \acmYear{2020}
% \acmMonth{8}
% \acmDOI{} % \acmDOI{10.1145/nnnnnnn.nnnnnnn}
% \startPage{1}

%% Copyright information
%% Supplied to authors (based on authors' rights management selection;
%% see authors.acm.org) by publisher for camera-ready submission;
%% use 'none' for review submission.
% \setcopyright{none}
%\setcopyright{acmcopyright}
%\setcopyright{acmlicensed}
%\setcopyright{rightsretained}
%\copyrightyear{2018}           %% If different from \acmYear

%%% The following is specific to ICFP '20 and the paper
%%% 'Kinds Are Calling Conventions'
%%% by Paul Downen, Zena M. Ariola, Simon Peyton Jones, and Richard A. Eisenberg.
%%%
\setcopyright{rightsretained}
\acmPrice{}
\acmDOI{10.1145/3408986}
\acmYear{2020}
\copyrightyear{2020}
\acmSubmissionID{icfp20main-p93-p}
\acmJournal{PACMPL}
\acmVolume{4}
\acmNumber{ICFP}
\acmArticle{104}
\acmMonth{8}

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

% \RequirePackage{xargs}

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

%% Style guide forbids boxes around figures
% \newenvironment{figurebox}{\begin{figure}\begin{framed}}{\end{framed}\end{figure}}
% \newenvironment{figurebox*}{\begin{figure*}\begin{framed}}{\end{framed}\end{figure*}}
\newenvironment{figurebox}{\begin{figure}}{\end{figure}}
\newenvironment{figurebox*}{\begin{figure*}}{\end{figure*}}

\crefname{figure}{Fig.}{Figs.}
\Crefname{figure}{Fig.}{Figs.}
\crefname{restriction}{Restriction}{Restrictions}

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

%if style == poly
%format checktype(e) = e
%format |-> = "\mapsto"
%format >=> = "\mathrel{{>}\!{=}\!{>}}"
%format >.> = "\mathrel{{>}\!{\circ}\!{>}}"
%format |> = "\triangleright"
%format /= = "\neq"

%format property name (vars) (lhs) (rhs) = "\forall" vars ", " lhs "\equiv" rhs
%format propertyImpl name (vars) (premise) (lhs) (rhs) = "\forall" vars ", " premise "\Rightarrow" lhs "\equiv" rhs

%format e1
%format e2
%format m1
%format m2
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
import qualified Bag
import Bag ( Bag )
import Data.Set ( Set)
import qualified Data.Set as Set
import RandomType
import Test.QuickCheck ( (==>) )

import GHC.TypeLits ( Nat )

class Dummy (n :: Nat) where
  method :: ()

f1 >=> f3 = \x -> do y <- f1 x
                     f3 y

(>.>) = flip (.)

(|>) = flip ($)

empty = undefined
lookup = undefined
alter = undefined
union = undefined
mapEM = undefined
foldEM = undefined

type Var = String

unionExprS = unionWithExprS const

data ListExprMap v
lookupListExpr = undefined
lookupList0 = undefined
lookupList1 = undefined

{-# NOINLINE exf #-}
exf :: Int -> Int -> Char
exf = undefined
exf2 :: Int -> Char
exf2 = undefined

insertMExpr = undefined
lookupMExpr = undefined
data MExprMap v

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
            _                          -> False
      go lhsmap rhsmap next (App a b) (App c d) = go lhsmap rhsmap next a c && go lhsmap rhsmap next b d
      go lhsmap rhsmap next (Lam v1 e1) (Lam v2 e2) = go (Map.insert v1 next lhsmap) (Map.insert v2 next rhsmap) (next+1) e1 e2
      go _ _ _ _ _ = False

\end{code}

%format checktype(e) = "instance Dummy $( randomType ) where method = const () (" e ")"

%endif


\begin{document}

\newcommand{\simon}[1]{[{\bf SLPJ}: {\color{red} #1}]}
\newcommand{\js}[1]{{\bf JS}: {\color{olive} #1} {\bf End JS}}
\newcommand{\rae}[1]{{\bf RAE}: {\color{dkblue} #1} {\bf End RAE}}
\newcommand{\sg}[1]{{\bf SG}: {\color{darkbrown} #1} {\bf End SG}}

\newcommand{\bv}[1]{\#_{#1}}    % Lambda-bound variable occurrence
\newcommand{\pv}[1]{\$_{#1}}    % Pattern variable binder
\newcommand{\pvo}[1]{\%_{#1}}   % Pattern variable occurrence

% Benchmark formatting hooks
\newcommand{\benchname}[1]{\texttt{#1}}
\newcommand{\insigdig}[1]{\ensuremath{\tilde{\text{#1}}}} % How to mark insignificant (within 2*σ) digits

%% Title information
\title%[Short Title]
{Triemaps that match}         %% [Short Title] is optional;
                                        %% when present, will be used in
                                        %% header instead of Full Title.
% \titlenote{with title note}             %% \titlenote is optional;
%                                         %% can be repeated if necessary;
%                                         %% contents suppressed with 'anonymous'
% \subtitle{Subtitle}                     %% \subtitle is optional
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
  \institution{Microsoft Research}
  % \streetaddress{21 Station Rd.}
  \city{Cambridge}
  % \postcode{CB1 2FB}
  \country{UK}
}
\email{simonpj@@microsoft.com}

\author{Richard A.~Eisenberg}
\affiliation{
  \institution{Tweag I/O}
  \city{Cambridge}
  \country{UK}
}
\email{rae@@richarde.dev}

\author{Sebastian Graf}
\affiliation{%
  \institution{Karlsruhe Institute of Technology}
  \city{Karlsruhe}
  \country{Germany}
}
\email{sebastian.graf@@kit.edu}

%% Abstract
%% Note: \begin{abstract}...\end{abstract} environment must come
%% before \maketitle command
\begin{abstract}
TrieMaps are great.
\end{abstract}

%% \maketitle
%% Note: \maketitle command must come after title commands, author
%% commands, abstract environment, Computing Classification System
%% environment and commands, and keywords command.
\maketitle


\section{Introduction} \label{sec:intro}

The designs of many programming languages include a feature where some concrete use of a construct is matched against a set of possible interpretations, where the possible interpretations might be defined in terms of variables to be instantiated at concrete usages. For example:
\begin{itemize}
\item Haskell's class instances work this way: the user defines instances (which may contain variables), and then concrete usage sites of class methods require finding an instance that applies.
\item Agda, Coq, and Idris all have implicit-argument features directly inspired by Haskell's class mechanism.
\item An extension to Haskell allows overloaded instances, where we select the most specific instance (that is, one that is a specialization of any other possible matching instance).
\item C++ and Java both support function overloading, where a usage site of a function must be matched against a choice of implementation. C++'s templates and Java's generics allow for variables to be used in the implementations. There are sometimes multiple implementations that match; both languages choose the most specific match (for an appropriate definition of "most specific").
\item C++ separately allows template specialization, where a templated definition can have concrete specializations. Once again, selection of these specializations depends on a "most-specific" relation appropriate to the case.
\item Scala? C\#?
\end{itemize}
Beyond features specified in a language's design, optimizations may require such a structure. For example, GHC's rewrite rules~\cite{rewrite-rules} requires a similar lookup to find a mapping from expressions to rules that may apply.

Our concern here is the efficient implementation of this matching
operation. That is, we wish to define a structure mapping keys to
arbitrary values. The keys are chosen from a type described by a
context-free grammar and thus comprise small trees. We assume a total
ordering on such trees. The challenge lies in the fact that we want
our structure to support wildcard variables, which represent any tree
at all. We will write such variables with Greek letters. Accordingly,
we can view a mapping |alpha ||-> v| as an infinite mapping, connecting
all possible trees to |v|, or we can have |Maybe alpha ||-> v2| map all
trees whose root is |Maybe| to the value |v2|. Combining these to |alpha
||-> v, Maybe alpha ||-> v2| would map all trees to |v|, except those
trees that have |Maybe| at the root, which map to |v2|. Accordingly,
looking up in our structure find the most specific
match.\footnote{This aspect of our design is unforced. We could also
return all possible matches, instead of selecting the most
specific. See \cref{sec:most-specific}.}

Our contributions are as follows:

\begin{itemize}
\item We describe the language-agnostic Variable Trie-Map (VTM) data
  structure, with the semantics as described above (and made more precise in
  \cref{sec:vtm}). The keys in a VTM support local bound variables via
  on-the-fly conversion to de Bruijn levels, necessary to support polymorphic
  types or $\lambda$-expressions. Looking up in a VTM is linear in the size of
  the key, regardless of the size of the VTM itself. The operations on a VTM
  are proved to uphold the sensible properties described in
  \cref{sec:vtm-properties}.
\item Some languages require not only the matching behavior above, but also a
  \emph{unification} operation, where we find not only keys that match, but
  keys that would match if variables in the looked-up tree were further
  instantiated. \Cref{sec:unification} describes how we extend VTMs to support
  unification as well as matching.
\item While VTMs are time-efficient, they can be space-inefficient.
  \Cref{sec:path-compression} describes an easy optimization which drastically
  reduces the memory footprint of VTMs whose keys share little structure in
  common. This optimization shows a 30\% geometric mean savings in the size of
  the structure GHC uses to look up rewrite rules.
\item Our work was motivated by quadratic behavior in GHC, observed when
  checking for instance consistency among imports. This became a problem in
  practice in Haskell's use at Facebook. We report on our implementation
  within GHC, showing that it achieves a 98\% speedup against the previous
  (admittedly naive) implementation of instance tables.
\end{itemize}

\section{The problem we address}
\begin{figurebox}
%{
%if style == poly
%format Map0 = Map
%format Dots = "\ldots"
%endif
\begin{code}
type XT v = Maybe v -> Maybe v

data Map0 k v = Dots  -- a finite map from keys of type |k| to values of type |v|
checktype(Map.empty      :: Map k v)
checktype(Map.insert     :: Ord k => k -> v -> Map k v -> Map k v)
checktype(Map.lookup     :: Ord k => k -> Map k v -> Maybe v)
checktype(Map.alter      :: Ord k => XT v -> k -> Map k v -> Map k v)
checktype(Map.unionWith  :: Ord k => (v->v->v) -> Map k v -> Map k v -> Map k v)
checktype(Map.size       :: Map k v -> Int)
checktype(Map.foldr      :: (v -> r -> r) -> r -> Map k v -> r)

data Bag v  -- An unordered collection of values |v|
checktype(Bag.empty      :: Bag v)
checktype(Bag.single     :: v -> Bag v)
checktype(Bag.union      :: Bag v -> Bag v -> Bag v)
checktype(Bag.map        :: (v1 -> v2) -> Bag v1 -> Bag v2)

infixr 1 >=>  -- Kleisli composition
(>=>) :: Monad m => (a -> m b) -> (b -> m c) -> a -> m c

infixr 1 >.>   -- Forward composition
(>.>)  :: (a -> b) -> (b -> c) -> a -> c

infixr 0 |>   -- Reverse function application
(|>)  :: a -> (a -> b) -> b
\end{code}
%}
\caption{API for library functions}
\label{fig:containers} \label{fig:library}
\end{figurebox}

Our general task is as follows: \emph{implement an efficient finite mapping
from keys to values, in which the key is a tree}.
For example, an |Expr| data type might be defined like this:
%{
%if style == newcode
%format Expr = "Expr0"
%format App = "App0"
%format Lam = "Lam0"
%format Var = "Var0"
%endif
\begin{code}
data Expr = App Expr Expr | Lam  Var Expr | Var Var
\end{code}
Here |Var| is the type of variables; these can be compared for
equality and used as the key of a finite map.  Its definition is not important
for this paper, but for the sake of concreteness,
you may wish to imagine it is simply a string:
\begin{code}
type Var = String
\end{code}
%}
The data type |Expr| is capable of representing expressions like |add x y| and
|\lambda x. add x y|. We will use this data type throughout the paper, because it
has all the features that occur in real expression data types: free variables like |add|,
represented by a |Var| node;
lambdas which can bind variables (|Lam|), and occurrences of those bound variables (|Var|);
and nodes with multiple children (|App|).  A real-world expression type would have
many more constructors, including literals, let-expressions and suchlike.

\rae{A little redundant with some intro stuff.}
A finite map keyed by such expressions is extremely useful.
GHC uses such a map during its common sub-expression
elimination pass, where the map associates an
expression with the identifier bound to that expression; if the same
expression occurs again, we can look it up in the map, and replace the
expression with the variable.
GHC also does many lookups based on \emph{types} rather than
\emph{expressions}.  For example, when implementing type-class
instance lookup, or doing type-family reduction, GHC needs a map whose
key is a type.  Both types and expressions are simply trees, and so are
particular instances of the general task.

\subsection{Lookup modulo matching} \label{sec:matching-intro}

Beyond just the basic finite maps we have described, our practical setting
in GHC demands more: we want to a lookup that does \emph{matching}.  GHC supports
so-called \emph{rewrite rules} \cite{rewrite-rules}, which the user can specify like this:
\begin{code}
prag_begin RULES "map/map" forall f g xs. map f (map g xs) = map (f . g) xs prag_end
\end{code}
This rule asks the compiler to rewrite any target expression that matches the shape
of the left-hand side (LHS) of the rule into the right-hand side
(RHS).  We use the term \emph{pattern} to describe the LHS, and \emph{target} to describe
the expression we are looking up in the map.
The pattern is explicitly quantified over the \emph{pattern variables}
(here |f|, |g|, and |xs|) that
can be bound during the matching process.  In other words, \emph{we seek a substitution
for the pattern variables that makes the pattern equal to the target expression}.
For example, if the program we are compiling contains the expression
|map double (map square nums)|, we would like to produce a substitution
|f ||-> double, g ||-> square, xs ||-> nums| so that the substituted RHS
becomes |map (double . square) nums|; we would replace the former expression
with the latter in the code under consideration.

Of course, the pattern might itself have bound variables, and we would
like to be insensitive to alpha-conversion for those. For example:
\begin{code}
prag_begin RULES "map/id"  map (\x -> x) = \y -> y prag_end
\end{code}
\simon{Can we typeset the opening and closing pragma brackets more nicely? Shorter dash for a start.}
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

\subsection{The interface of of a finite map} \label{sec:interface}

What API might such a map have? Building on the design of widely
used functions in Haskell (see \cref{fig:containers}), we
seek these basic operations:
\begin{code}
emptyEM   :: ExprMap v
lookupEM  :: Expr -> ExprMap v -> Maybe v
alterEM   :: Expr -> XT v -> ExprMap v -> ExprMap v
\end{code}
The functions |emptyEM| and |lookupEM| should be
self-explanatory.  The function |alterTM| is a standard
generalisation of |insertEM|: instead of providing just
a new element to be inserted, the caller provides a
\emph{transformation} |XT v|, an
abbreviation for |Maybe v -> Maybe v| (see \Cref{fig:library}).  This function
transforms the existing value associated with key, if any (hence the
input |Maybe|), to a new value, if any (hence the output |Maybe|).
These fundamental operations on a finite map must obey these properties:
\begin{code}
property propLookupEmpty (e)                       (lookup e empty             ^^^^)  (Nothing)
property propLookupAlter (e m xt)                  (lookup e (alter e xt m)    ^^^^)  (xt (lookup e m))
propertyImpl propWrongElt (e1 e2 m xt) (e1 /= e2)  (lookup e1 (alter e2 xt m)  ^^^^)  (lookup e1 m)
\end{code}

We can easily define |insertEM| and |deleteEM| from |alterEM|:
\begin{code}
insertEM :: Expr -> v -> ExprMap v -> ExprMap v
insertEM e v = alterEM e (\_ -> Just v)

deleteEM :: Expr -> ExprMap v -> ExprMap v
deleteEM e = alterEM e (\_ -> Nothing)
\end{code}
You might wonder whether, for the purposes of this paper, we could just define |insert|,
leaving |alter| for the Appendix, but as we will see in \Cref{sec:alter}, our
approach using tries fundamentally requires the generality of |alter|.

We would also like to support other standard operations on finite maps, including
\begin{itemize}
\item An efficient union operation to combine two finite maps into one:
\begin{code}
unionEM :: ExprMap v -> ExprMap v -> ExprMap v
\end{code}
\item A map operation to apply a function to the range of the finite map:
\begin{code}
mapEM :: (a -> b) -> ExprMap a -> ExprMap b
\end{code}
\item A fold operation to combine together the elements of the range:
\begin{code}
foldEM :: (a -> b -> b) -> ExprMap a -> b -> b
\end{code}
\end{itemize}

\subsection{Keys that include binders: alpha-renaming}

Recall that the type |Expr| is the \emph{key} of our |ExprMap| type.
We do not want our programming language to distinguish between the expressions
|\x -> x| and |\y -> y|, and so we would expect insertion and lookup to be insensitive to
$\alpha$-renaming. That is, the correctness properties we list above
must use $\alpha$-equivalence when comparing expressions for equality.

Making insert and lookup work modulo $\alpha$-equivalence is not difficult
in priciple; for example, we the key expression to one using de Bruijn levels or indices,
and then work with that. However, doing so requires making a copy of the input
key, a potentially expensive extra step. We will see that it can be accommodated
with a simple on-the-fly indexing.
\rae{Remind me why hashing-modulo-alpha is hard, referring to
  \cite{alpha-hashing}.}\simon{Not relevant here; our ``hashing modulo alpha paper
  is only relevant if you want
  compositional hashing of every node.}

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
While this double-check is not so terrible, we will see that
the naive approach described here does not extend well to support
the extra features we require in our finite maps.

But the killer is this: \emph{neither binary search trees nor hashing is compatible
with matching lookup}.  For our purposes they are non-starters.

What other standard solutions are there, apart from linear search?
The theorem proving and automated reasoning community has been working with huge sets
of rewrite rules, just as we describe, for many years.
They have developed \emph{discriminator trees} for this job, which
embody essentially the same core ideas as those we present below.  But there are many
differences, as we discuss later in \Cref{sec:related}.

\section{Tries} \label{sec:Expr}

A standard approach to a finite map in which the key has internal structure
is to use a \emph{trie}\footnote{\url{https://en.wikipedia.org/wiki/Trie}}.
Let us consider a simplified
form of expression:
\begin{code}
data Expr = Var Var | App Expr Expr
\end{code}
We leave lambdas out for now,
so that all |Var| nodes represent free variables, which are treated as constants.
We will return to lambdas in \Cref{sec:binders}.

\subsection{The basic idea} \label{sec:basic}

Here is a trie-based implemenation for |Expr|:
%{
%if style == newcode
%format ExprMap = "ExprMap0"
%format EM = "EM0"
%format em_var = "em_var0"
%format em_app = "em_app0"
%format lookupExpr = "lookupExpr0"
%format alterExpr = "alterExpr0"
%format liftXT = "liftXT0"
%format emptyExpr = "emptyExpr0"
%endif
\begin{code}
data ExprMap v = EM  { em_var  :: Map Var v, em_app  :: ExprMap (ExprMap v) }
\end{code}
Here |Map Var v| is any standard, existing finite map, such as the |containers|
library\footnote{\url{https://hackage.haskell.org/package/containers}
} keyed by |Var|, with values |v|.
One way to understand this slightly odd data type is to study its lookup function:
\begin{code}
lookupExpr :: Expr -> ExprMap v -> Maybe v
lookupExpr e (EM { em_var = m_var, em_app = m_app })
  =  case e of
      Var x      -> Map.lookup x m_var
      App e1 e2  ->  case lookupExpr e1 m_app of
                        Nothing  -> Nothing
                        Just m1  -> lookupExpr e2 m1
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
%format lookupExpr = "lookupExpr2"
\begin{code}
lookupExpr :: Expr -> ExprMap v -> Maybe v
  -- we need this type signature because the body is polymorphic recursive
\end{code}
%endif
\begin{code}
lookupExpr (Var x)      = em_var  >.> Map.lookup x
lookupExpr (App e1 e2)  = em_app  >.> lookupExpr e1 >=> lookupExpr e2
\end{code}
%}
The function |em_var :: ExprMap v -> Map Var v|
is the auto-generated selector that picks |em_var| field from an |EM| record, and similarly |em_app|.
The functions |(>.>)| and |(>=>)| are right-associative forward composition
operators, respectively monadic and non-monadic,
that chain the individual operations together (see Cref{fig:library}).
Finally, we have $\eta$-reduced the definition, by omitting the |m| parameter.
These abbreviations become quite worthwhile when we add more constructors, each with more fields,
to the key data type.

Notice that in contrast to the approach of \Cref{sec:ord}, \emph{we never compare two expressions
for equality or ordering}.  We simply walk down the |ExprMap| structure, guided
at each step by the next node in the target.  (We typically use the term ``target'' for the
key we are looking up in the finite map.)

This definition is extremely short and natural. But it conceals a hidden
complexity: \emph{it requires polymorphic recursion}. The recursive call to |lookupExpr e1|
instantiates |v| to a different type than the parent function definition.
Haskell supports polymorphic recurision readily, provided you give type signature to
|lookupExpr|, but not all languages do.

\subsection{Modifying tries} \label{sec:alter} \label{sec:empty-infinite}

It is not enough to look up in a trie -- we need to \emph{build} them too!
First, we need an empty trie. Here is one way to define it:
%{
%if style == newcode
%format emptyExpr = "emptyExpr0"
%format foldrExpr = "foldrExpr0"
%format sizeExpr = "sizeExpr0"
%format ExprMap = "ExprMap0"
%format em_var = "em_var0"
%format em_app = "em_app0"
%format EM = "EM0"
%endif
\begin{code}
emptyExpr :: ExprMap v
emptyExpr = EM { em_var = Map.empty, em_app = emptyExpr }
\end{code}
It is interesting to note that |emptyExpr| is an infinite, recursive structure:
the |em_app| field refers back to |emptyExpr|.  We will change this
definition in \Cref{sec:empty}, but it works perfectly well for now.

Next, we need to |alter| a triemap:
\begin{code}
alterExpr :: Expr -> XT v -> ExprMap v -> ExprMap v
alterExpr e xt m@(EM { em_var = m_var, em_app = m_app })
  =  case e of
       Var x      -> m { em_var  = Map.alter xt x m_var }
       App e1 e2  -> m { em_app  = alterExpr e1 (liftXT (alterExpr e2 xt)) m_app }

liftXT :: (ExprMap v -> ExprMap v) -> XT (ExprMap v)
liftXT f Nothing    = Just (f emptyExpr)
liftXT f (Just m)   = Just (f m)
\end{code}
%}
In the |Var| case, we must just update the map stored in the |em_var| field,
using the |Map.alter| function from \Cref{fig:containers};
in Haskell the notation ``|m { fld = e }|'' means the result
of updating the |fld| field of record |m| with new value |e|.
In the |App| case we look up |e1| in |m_app|;
we should find a |ExprMap| there, which we want to alter with |xt|.
We can do that with a recursive call to |alterExpr|, using |liftXT|
for impedence-matching.

The |App| case shows why we need the generality of |alter|.
Suppose we attempted to define an apparently-simpler |insert| operations.
Its equation for |(App e1 e2)| would look up |e1| --- and would then
need to \emph{alter} that entry (an |ExprMap|, remember) with the result of
inserting |(e2,v)|.  So we are forced to define |alter| anyway.

We can abbreviate the code for |alterExpr| using combinators, as we did in the case of
lookup, and doing so pays dividends when the key is a data type with
many constructors, each with many fields.  However, the details are
fiddly and not illuminating, so we omit them here.  Indeed, for the
same reason, in the rest of this paper we will typically omit the code
for |alter|, though the full code is available in the
Appendix.

\subsection{Unions of maps}

A common operation on finite maps is to take their union:
\begin{code}
unionExpr :: ExprMap v -> ExprMap v -> ExprMap v
\end{code}
In tree-based implementations of finite maps, such union operations can be tricky.
The two trees, which have been built independently, might not have the same
left-subtree/right-subtree structure, so some careful re-alignment may be required.
But for tries there are no such worries --
their structure is identical, and we can simply zip them together.  There is one
wrinkle: just as we had to generalise |insert| to |alter|,
to accomodate the nested map in |em_app|, so we need to generalise |union| to |unionWith|:
\begin{code}
unionWithExpr :: (v -> v -> v) -> ExprMap v -> ExprMap v -> ExprMap v
\end{code}
When a key appears on both maps, the combining function is used to
combine the two corresponding values.
With that generalisation, the code is as follows:
\begin{code}
unionWithExpr f  (EM { em_var = m1_var, em_app = m1_app })
                 (EM { em_var = m2_var, em_app = m2_app })
  = EM  { em_var = Map.unionWith f m1_var m2_var
         , em_app = unionWithExpr (unionWithExpr f) m1_app m2_app }
\end{code}
It could hardly be simpler.

\subsection{Folds and the empty map} \label{sec:fold} \label{sec:empty}

This strange, infinite definition of |emptyExpr| given in \Cref{sec:empty-infinite}
works fine (in a lazy language at least) for lookup, alteration, and union, but it fails
fundamentally when we want to \emph{iterate} over the elements of the trie.
For example, suppose we wanted to count the number of elements in the finite map; in |containers|
this is the function |Map.size| (\Cref{fig:containers}).  We might attempt:
%{
%if style == poly
%format undefined = "???"
%endif
\begin{code}
sizeExpr :: ExprMap v -> Int
sizeExpr (EM { em_var = m_var, em_app = m_app }) = Map.size m_var + undefined
\end{code}
%}
We seem stuck because the size of the |m_app| map is not what we want: rather,
we want to add up the sizes of its \emph{elements}, and we don't have a way to do that yet.
The right thing to do is to generalise to a fold:
\begin{code}
foldrExpr :: (v -> r -> r) -> r -> ExprMap v -> r
foldrExpr k z (EM { em_var = m_var, em_app = m_app })
  = Map.foldr k z1 m_var
  where
    z1 = foldrExpr kapp z m_app
    kapp m1 r = foldrExpr k r m1
\end{code}
%}
In the binding for |z1| we fold over |m_app :: ExprMap (ExprMap v)|.
The function |kapp| is combines the map we find with the accumulator, by again
folding over the map with |foldrExpr|.

But alas, |foldrExpr| will never terminate!  It always invokes itself immediately
(in |z1|) on |m_app|; but that invocation will again recursively invoke
|foldrExpr|; and so on for ever.
The solution is simple: we just need an explicit representation of the empty map.
Here is one way to do it (we will see another in \Cref{sec:generalised}):
%{
%if style == newcode
%format ExprMap = "ExprMap1"
%format EmptyEM = "EmptyEM1"
%format EM = "EM1"
%format em_var = "em_var1"
%format em_app = "em_app1"
%endif
\begin{code}
data ExprMap v  = EmptyEM
                | EM { em_var :: Map Var v, em_app :: ExprMap (ExprMap v) }

emptyExpr :: ExprMap v
emptyExpr = EmptyEM

foldrExpr :: (v -> r -> r) -> r -> ExprMap v -> r
foldrExpr k z EmptyEM                                   = z
foldrExpr k z (EM { em_var = m_var, em_app = m_app })  = Map.foldr k z1 m_var
  where
    z1 = foldrExpr kapp z m_app
    kapp m1 r = foldrExpr k r m1
\end{code}
Equipped with a fold, we can easily define the size function, and another
that returns the range of the map:\simon{Typesetting trouble: need a space between underscore and n}
\begin{code}
sizeExpr :: ExprMap v -> Int
sizeExpr = foldrExpr (\_ n -> n+1) 0

elemsExpr :: ExprMap v -> [v]
elemsExpr = foldrExpr (:) []
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
   emptyTM   :: tm a
   lookupTM  :: TrieKey tm -> tm a -> Maybe a
   alterTM   :: TrieKey tm -> XT a -> tm a -> tm a
   foldTM    :: (a -> b -> b) -> tm a -> b -> b
   unionTM   :: tm a -> tm a -> tm a
   dots
\end{code}
%}
The class constraint |TrieMap tm| says that the type |tm| is a triemap, with operations
|emptyTM|, |lookupTM| etc.
The class has an \emph{associated type} \cite{associated-types}, |TrieKey tm|,
a type-level function that transforms the type of the triemap into
the type of \emph{keys} of that triemap.

Now we can witness the fact that |ExprMap| is a |TrieMap|, like this:
\rae{Are there definitions for |alterTM| and |foldTM| for this, if we wanted them?}
\simon{Yes, in GenTrieMap.hs. But we need an Appendix that sets it all out.}
%{
%if style == poly
%format dots = "\ldots"
%else
%format dots = "foldTM = undefined"
%endif
\begin{code}
instance TrieMap ExprMap where
  type TrieKey ExprMap = Expr
  emptyTM   = emptyExpr
  lookupTM  = lookupExpr
  alterTM   = alterExpr
  dots
\end{code}
%}
Having a class allow us to write helper functions that work for any triemap,
such as
\begin{code}
insertTM :: TrieMap tm => TrieKey tm -> v -> tm v -> tm v
insertTM k v = alterTM k (\_ -> Just v)

deleteEM :: TrieMap tm => TrieKey tm -> tm v -> tm v
deleteEM k = alterEM k (\_ -> Nothing)
\end{code}
But that is not all.
Suppose our expressions had multi-argument apply nodes, |AppV|, thus
%{
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
Then we would need to built a trie keyed by a \emph{list} of |Expr|.
A list is just another algebraic data type, built with nil and cons,
so we \emph{could} use exactly the same approach, thus
\begin{code}
lookupListExpr :: [Expr] -> ListExprMap v -> Maybe v
\end{code}
But rather than define a |ListExprMap| for keys of type |[Expr]|,
and a |ListDeclMap| for keys of type |[Decl]|, etc, we would obviously prefer
to build a trie for lists of \emph{any type}, like this \cite{hinze}:
\begin{code}
lookupList :: TrieMap tm => [TrieKey tm] -> ListMap tm v -> Maybe v
lookupList []      = lm_nil
lookupList (k:ks)  = lm_cons >.> lookupTM k >=> lookupList ks

emptyList :: TrieMap tm => ListMap tm
emptyList = LM { lm_nil = Nothing, lm_cons = emptyTM }

data ListMap tm v = LM { lm_nil  :: Maybe v, lm_cons :: tm (ListMap tm  v) }
\end{code}
The code for |alterList| and |foldList| is routine. Notice that all of
these functions are polymorphic in |tm|, the triemap for the list elements.
So |ListMap| is a \emph{triemap-transformer}; and if |tm| is a |TrieMap| then
so is |ListMap tm|:
\begin{code}
instance TrieMap tm => TrieMap (ListMap tm) where
   type TrieKey (ListMap tm) = [TrieKey tm]
   emptyTM  = emptyList
   lookupTM = lookupList
   ...
\end{code}

\subsection{Singleton maps, and empty maps revisited} \label{sec:singleton}

Suppose we start with an empty map, and insert a value
with a key (an |Expr|) that is large, say
\begin{spec}
  App (App (Var "f") (Var "x")) (Var "y")
\end{spec}
Looking at the code
for |alterExpr| in \Cref{sec:alter}, you can see that
because there is an |App| at the root, we will build an
|EM| record with an empty |em_var|, and an
|em_app| field that is... another |EM|
record.  Again the |em_var| field will contain an
empty map, while the |em_app| field is a further |EM| record.

In effect, the key is linearised into a chain of |EM| records.
This is great when there are a lot of keys with shared structure, but
once we are in a sub-tree that represents a single key-value pair it is
a rather inefficient way to represent the key.  So a simple idea is this:
when a |ExprMap| represents a single key-value pair, represent it
as directly a key-value pair, like this:
\begin{code}
data ExprMap v  = EmptyEM
                | SingleEM Expr v   -- A single key/value pair
                | EM { em_var :: Map Var v, em_app :: ExprMap (ExprMap v) }
\end{code}
But we will have to tiresomely repeat these extra data constructors, |EmptyX| and |SingleX|
for each new data type |X| for which we want a triemap.
For example we would have to add |EmptyList| and |SingleList| to the |ListMap| data type
of \Cref{sec:class}.
It is better instead to abstract over the enclosed triemap, like this:
\begin{code}
data SEMap tm v  = EmptySEM
                 | SingleSEM (TrieKey tm) v
                 | MultiSEM  (tm v)
\end{code}
The code for lookup practically writes itself:
\begin{code}
lookupSEMap :: TrieMap tm => TrieKey tm -> SEMap tm v -> Maybe v
lookupSEMap _   EmptySEM                       = Nothing
lookupSEMap tk  (SingleSEM pk v)  | tk == pk   = Just v
                                  | otherwise  = Nothing
lookupSEMap tk  (MultiSEM tm)                  = lookupTM tk tm
\end{code}
Notice that in the |SingleSEM| case we need equality on the key type |TrieKey tm|,
to tell if the key being looked up, |tk| is the same as the key in
the |SingleEM|, namely |pk|.
That is why we made |Eq (TrieKey tm)| a superclass of |TrieMap tm|
in the |class| declaration in \Cref{sec:class}.

The code for alter is more interesting, becuase it governs the shift from
|EmptySEM| to |SingleSEM| and thence to |MultiSEM|:
\begin{code}
alterSEM :: TrieMap tm => TrieKey tm -> XT v -> SEMap tm v -> SEMap tm v
alterSEM k xt EmptySEM = case xt Nothing of  Nothing  -> EmptySEM
                                             Just v   -> SingleSEM k v
alterSEM k1 xt (SingleSEM k2 v2)
  | k1 == k2   = case xt (Just v2) of
                      Nothing  -> EmptySEM
                      Just v'  -> SingleSEM k2 v'
  | otherwise  = case xt Nothing of
                      Nothing  -> SingleSEM k2 v2
                      Just v1  -> MultiSEM (insertTM k1 v1 (insertTM k2 v2 emptyTM))
alterSEM k xt (MultiSEM tm) = MultiSEM (alterTM k xt tm)
\end{code}
Now, of course, we can make |SEMap| itself an instance of |TrieMap|:
\begin{code}
instance TrieMap tm => TrieMap (SEMap tm) where
  type TrieKey (SEMap tm) = TrieKey tm
  emptyTM  = EmptySEM
  lookupTM = lookupSEM
  alterTM  = alterSEM
  foldTM   = foldSEM
\end{code}
Adding a new item to a triemap can turn |EmptySEM| into |SingleSEM| and |SingleSEM|
into |MultiSEM|; and deleting an item from a |SingleSEM| turns it back into |EmptySEM|.
But you might wonder whether we can shrink a |MultiSEM| back to a |SingleSEM| when it has
only one remaining element?
Yes, of course we can, but it takes a bit of work; the Appendix has the details.

Finally, we need to re-define |ExprMap| and |ListMap| using |SEMap|:
\begin{code}
  type ExprMap       = SEMap ExprMap'
  data ExprMap' v    = EM { em_var  :: Map Var v, em_app  :: ExprMap (ExprMap v) }

  type ListMap        = SEMap ListMap'
  data ListMap' tm v  = LM { lm_nil  :: Maybe v, lm_cons :: tm (ListMap tm v) }
\end{code}
The auxiliary data types |ExprMap'| and |ListMap'| have only a single constructor, because
the empty and singleton cases are dealt with by |SEMap|.  We reserve the original,
un-primed, names for the user-visible |ExprMap| and |ListMap| constructors.

The singleton-map optimisation makes a big difference in practice: see \Cref{sec:results}.

\subsection{Generic programming}

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

Thus far we have usefully consolidated the state of the art, but have not really done
anything new.  Tries are well known, and there are a number of papers about
tries in Haskell \cite{hinze etc}.  However, none of these works deal with keys that contain
binders, and that should be insensitive to alpha-conversion.  That is the challenge we
address next.  Here is our data type, |ExprL|, where the ``L'' connotes the new |Lam| constructor:
\begin{code}
data ExprL = AppL ExprL ExprL | Lam Var ExprL | VarL Var
\end{code}
The key idea is simple: we perform de-Bruijn numbering on the fly,
renamign each binder to a natural number, from outside in.
So, when inserting or looking up a key $(\lambda x.\, foo~ (\lambda y.\, x+y))$ we
behave as if the key was $(\lambda.\, foo ~(\lambda. \bv{1} + \bv{2}))$, where
each $\bv{i}$ stands for an occurrence of the variable bound by the $i$'th lambda.
In effect, then, we behave as if the data type was like this:
\begin{spec}
data Expr' = AppL ExprL ExprL | Lam ExprL | FreeVar Var | BoundVar Int
\end{spec}
Notice (a) the |Lam| node no longer has a binder and (b) there are
two sorts of |VarL| nodes, one for free variables and one for bound
variables. We will not actually build a value of type |Expr'| and look
that up in a trie keyed by |Expr'|; rather,
we are going to \emph{behave as if we did}. Here is the code
\begin{code}
data ExprLMap v = ELM  {  elm_app  :: ExprLMap (ExprLMap v)
                       ,  elm_lam  :: ExprLMap v
                       ,  elm_fv   :: Map Var v            -- Free variables
                       ,  elm_bv   :: Map BoundVarKey v }  -- Lambda-bound variables

lookupExprL :: ExprL -> ExprLMap v -> Maybe v
lookupExprL e = lkExprL (DB emptyBVM e)

data DBExprL = DB { edb_bvm :: BoundVarMap, edb_expr :: ExprL }

lkExprL :: DBExprL -> ExprLMap v -> Maybe v
lkExprL (DB bvm (AppL e1 e2)) = elm_app >.> lkExprL (DB bvm e1) >=> lkExprL (DB bvm e2)
lkExprL (DB bvm (Lam v e))   = elm_lam >.> lkExprL (DB (extendBVM v bvm) e)
lkExprL (DB bvm (VarL v))     = case lookupBVM v bvm of
                                Nothing -> elm_fv  >.> Map.lookup v   -- Free
                                Just bv -> elm_bv  >.> Map.lookup bv  -- Lambda-bound

data BoundVarMap = BVM { bvm_next :: BoundVarKey, bvm_map :: Map Var BoundVarKey }
type BoundVarKey = Int

emptyBVM :: BoundVarMap
emptyBVM = BVM { bvm_next = 1, bvm_map = Map.empty }

extendBVM :: Var -> BoundVarMap -> BoundVarMap
extendBVM v (BVM { bvm_next = n, bvm_map = bvm })
  = BVM { bvm_next = n+1, bvm_map = Map.insert v n bvm }

lookupBVM :: Var -> BoundVarMap -> Maybe BoundVarKey
lookupBVM v (BVM {bvm_map = bvm }) = Map.lookup v bvm
\end{code}
We maintain a |BoundVarMap|
that maps each lambda-bound variable to its de-Bruijn level\footnote{
  The de-Bruijn \emph{index} of the occurrence of a variable $v$ counts the number
  of lambdas between the occurrence of $v$ and its binding site.  The de-Bruijn \emph{level}
  of $v$ counts the number of lambdas between the root of the expression and $v$'s binding site.
  It is convenient for us to use \emph{levels}.
  \simon{What can we cite?  Sebastian had a couple of suggestions, but both are informal.}
  \sg{We should cite the original deBruijn paper. See Figure 1 here, which is explained just below:
      \url{http://alexandria.tue.nl/repository/freearticles/597619.pdf}
      reference depth corresponds to DB index. But it appears that the DB level is defined entirely different.}
  }, of type |BoundVarKey|.
% \sg{We are using de Bruijn \emph{levels}, not \emph{indices}. A Var occurrence
% with De Bruijn indices would count the number of lambdas between the occ and
% its binding site. Levels, on the other hand, count the number of lambdas from
% the root of the expr to the binding site. So within the const function
% $\lambda x. \lambda y. y$, the occ of $y$ has DB index 0, but DB level 1.
% Indices are favorable when doing subsitution which I presume we don't. See
% also \url{https://randall-holmes.github.io/Watson/babydocs/node26.html} and
% \url{https://mail.haskell.org/pipermail/haskell-cafe/2007-May/025424.html}.}
The key we look up --- the first argument of |lkExprL| --- becomes
a |DBExprL|, which is a pair of a |BoundVarMap| and an
|ExprL|.
At a |Lam|
node we extend the |BoundVarMap|. At a |Var| node we
look up the variable in the |BoundVarMap| to decide whether it is
lambda-bound (within the key) or free, and behave appropriately.
The code for |alter| and |fold| holds no new surprises.
The construction of \Cref{sec:generalised}, to handle empty and singleton maps,
applies without difficulty to this generalised map.

And that is really all there is to it: it is remarkably easy to extend the basic
trie idea to be insensitive to alpha-conversion.

\section{Tries that match} \label{sec:matching}

A key advantage of tries over hash-maps and balanced trees is
that they can naturally extend to support \emph{matching} (\Cref{sec:matching-intro}).
In this section we explain how.

\subsection{What ``matching'' means} \label{sec:matching-spec}

First, we have to ask what the API should be.
Our overall goal is to build a \emph{matching trie} into which we can:
\begin{itemize}
\item \emph{Insert} (pattern, value) pairs
\item \emph{Look up} a target expression, and return all the values whose pattern \emph{matches} that expression.
\end{itemize}
Semantically, then, a matching trie can be thought of as a set of \emph{entries},
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
For example, suppose we wanted to encode the rewrite rule
%{
%if style == newcode
%format f = "exf"
%format f2 = "exf2"
%endif
\begin{code}
prag_begin RULES "foo" forall x. f x x = f2 x prag_end
\end{code}
%}
Here the pattern $([x], f~ x~ x)$ has a repeated variable $x$,
and should match targets like $(f~ 1~ 1)$ or $(f ~(g~ v)~ (g ~v))$,
but not $(f~ 1~ (g~ v))$.  This ability is important if we are to use matching tries
to implement class or type-family look in GHC.

It is sometimes desirable to be able to look up the \emph{most specific match} in the matching trie.
For example, suppose the matching trie contains the following two (pattern,value) pairs:
$$
\{ ([a],\, f\, a),\;\; ([p,q],\, f\,(p+q)) \}
$$
and suppose we look up $(f\,(2+x))$ in the trie.  The first entry matches, but the second also matches (with $S = [p \mapsto 2, q \mapsto x]$), and \emph{the second pattern is a substitution instance of the first}.  In some applications
we may want to return just the second match.  We call this \emph{most-specific matching}.

\subsection{The API of a matching trie} \label{sec:match-api}

Here are the signatures of the lookup and insertion\footnote{We begin with |insert|
  because it is simpler than |alter|} functions for our new matching triemap, |MExprMap|:
\begin{code}
type ExprPat = ([PatVar], Expr)
type PatVar  = Var
type Match v = ([(PatVar, Expr)], v)

insertMExpr :: ExprPat -> v -> MExprMap v -> MExprMap v
lookupMExpr :: Expr -> MExprMap v -> Bag (Match v)
\end{code}
\rae{What is the actual definition of |MExprMap|?}
A |MExprMap| is a trie, keyed by |ExprPat| \emph{patterns}.
A pattern variable, of type |PatVar| is just a |Var|; we
use the type synonym just for documentation purposes. When inserting into a
|MExprMap| we supply a pattern expression paired with the |[PatVar]|
over which the pattern is quantified.  When looking up in the map we return a \emph{bag}
of results (because more than one pattern might match).  Each item in this bag is
a |Match| that includes the |(PatVar, Expr)| pairs obtained by
matching the pattern, plus the value in the map (which presumably mentions those
pattern variables).

A |Bag| is a standard un-ordered collection of values, with a union operation;
see \Cref{fig:library}. We need to be able to return a bag because there may
be multiple matches. Even if we are returning the most-specific matches,
there may be multiple incomparable ones.

% \sg{Why does lookupMExpr return a Bag? I thought we care for most-specific
% matches? Shouldn't it then return a DAG of matches, or a tree, or at least a
% list? Bag means no order at all... Later code assumes we can call map on Bags,
% but Bag isn't defined anywhere. Maybe just return a list?
% (A long time later, after I read 5.7) Ah, so it really is unordered. Fair
% enough, but it would help to say as much.}
% \simon{Any better now?}
% Yes.

\subsection{Canonical patterns and pattern keys}

In \Cref{sec:binders} we saw how we could use de-Bruijn levels to
make two lambda expressions that differ only superficially (in the
name of their bound variable) look the same.  Clearly, we want to do
the same for pattern variables.  After all, consider these two patterns:
$$
([a,b], f~a~b~True) \qquad and \qquad ([p,q], f~q~p~False)
$$
The two pattern expressions share a common prefix, but differ both in the
\emph{names} of the pattern variable and in their \emph{order}. We might hope
to suppress the accidental difference of names by using numbers instead -- we will
use the term \emph{pattern keys} for these numbers.
But from the set of pattern variables alone, we
cannot know \emph{a priori} which key to assign to which variable.

Our solution is to number the pattern variables \emph{in order of their
first occurrence in a left-to-right scan of the expression}\footnote{As we shall
  see, this is very convenient in implementation terms.}.
As in \Cref{sec:binders} we will imagine that we cannicalise the pattern, although
in reality we will do so on-the-fly, without ever constructing the cannonicalised pattern.
Be that as it may, the canonicalised patterns become:
$$
   f~\pv{1}~\pv{2}~True      \qquad and \qquad  f~\pv{1}~\pv{2}~False
$$
By numbering the variables left-to-right, we ensure that they ``line up''.
In fact, since the pattern variables are numbered left-to-right we don't even
need the subscripts (just as we don't need a subscript on the lambda in
de-Bruijn notation), so the canonicalised patterns become
$$
   f~\pv{}~\pv{}~True      \qquad and \qquad  f~\pv{}~\pv{}~False
$$

\sg{I found that point \emph{very} confusing. In contrast to binding occs in
lambdas, those occs are not binding at all! Their binding site is in the list of
pattern vars. Hence it is completely obvious to me that we need the subscript.
The following paragraph comes to the same conclusion, but justifies it by saying
that the first occ of every pattern var is the binding occ. I mean, I see
that we \emph{can} omit the index on $\pv{}$, because it is redundant by the
numbering we propose, but \emph{should} we really omit it? It's probably
important for identifying whether the PV is flexible or rigid, fair enough...

Why is the current design better than the simple and obvious solution where we
label all $\pv{}$ with an index and don't bother having $\pvo{}$ altogether?
Handling flex-rigid vs. rigid-rigid constraints is pretty standard in
unification literature... Why can't we do it similarly?}

What if the variable occurs more than once? For example, suppose we are matching
the pattern $([x],\, f\, x\,x\,x)$ against the target expression
$(f\,e_1\,e_2\,e_3)$.  At the first occurrence of the pattern variable $x$
we succeed in matching, binding $x$ to $e_1$; but at the second
occurrence we must note that $x$ has already been bound, and instead
check that $e_1$ is equal to $e_2$; and similarly at the third occurrence.
These are very different actions,
so it is helpful to distinguish the first occurrence from subsequent
ones when canonicalising.  So our pattern $([x],\, f\, x\,x\,x)$ might
be canonicalised to $(f\,\pv{}\,\pvo{1}\,\pvo{1})$, where the first (or binding) occurrence
is denoted $\pv{}$ and subsequent (bound) occurrences of pattern variable $i$ are denoted $\pvo{i}$.

For pattern-variable occurrences we really do need the subcript! Consider the
patterns $$([x,y], f\,x\,y\,y\,x) \qquad and \qquad ([p,q], f\,q\,p\,q\,p)$$
which differ not only in the names of their pattern variables, but also in the
order in which they occur in the pattern.
They canonicalise to
$$(f \,\pv{}\, \pv{}\, \pvo{2}\, \pvo{1}) \qquad and  \qquad (f \,\pv{}\, \pv{}\, \pvo{1}\, \pvo{2})$$
respectively.  The subscripts are essential to keep these two patterns distinct.

\subsection{Undoing the pattern keys} \label{sec:patkeymap}

The trouble with canonicalising our patterns (to share the structure of the patterns)
is that matching will produce a substitution mapping patttern \emph{keys} to
expressions, rather that mapping pattern \emph{variables} to values.  For example,
suppose we start with the pattern $([x,y], f \,x\, y\, y\, x)$ from the
end of the last section. Its canonical form is $(f \,\pv{}\, \pv{}\, \pvo{2}\, \pvo{1})$.
If we match that against a target $(f\,e_1\,e_2\,e_2\,e_1)$ we will produce a substitution $[\pvo{1} \mapsto e_1, \pvo{2} \mapsto e_2]$.
But  what we \emph{want} is a |Match| (\Cref{sec:match-api}),
that gives a list of (pattern-variable, expression) pairs $[(x, e_1), (y,e_2)]$.
\sg{What is the difference between ,,expression'' and ,,value''
here? I think for a Match, you have to give the value in addition to the list of
pairs.}\simon{True, we should say ``expression'' here; fixed.  And indeed a Match includes the value
as well as the substitution, see 5.2. I'm not sure what to say to make this clearer.}

Somehow we must accumulate a \emph{pattern-key map} that, for each
individual entry in triemap, maps its pattern keys back to the corresponding
pattern variables for that entry.  The pattern-key map is just a list of (pattern-variable, pattern-key) pairs.
For our example the pattern key map would be
$[(x, \pv{1}), (y,\pv{2})]$.  We can store the pattern key
map paired with the value, in the triemap itself,
so that once we find a successful match we can use the pattern
key map and the pattern-key substitution to recover the pattern-variable substition that we want.

To summarise, suppose we want to build a matching trie for the following (pattern, value) pairs:
$$
(([x,y],\; f\;y\;(g\;y\;x)),\; v_1) \qquad and \qquad (([a],\; f\;a\;True),\;v_2)
$$
Then we will build a trie withe the following entries (key-value pairs):
$$
( (f \;\pv{}\;(g\;\pvo{1}\;\pv{})),\; ([(x,\pv{2}),(y,\pv{1})], v_1) )
  \qquad and \qquad
( (f \;\pv{}\;True),\; ([(a,\pv{1})],\;v_2) )
$$


\subsection{Implementation: lookup} \label{sec:matching-lookup}

We are finally ready to give an implementation of matching tries.
We begin with |Expr| (defined in \Cref{sec:Expr}) as our key type;
that is we will not deal with lambdas and lambda-bound variables for now.
\Cref{sec:binders} will apply with no difficulty, but we can add that back
in after we have dealt with matching.
With these thoughts in mind, our matching trie has this definition:
\begin{code}
type PatKeys     = [(PatVar,PatKey)]
type MExprMap v = MExprMapX (PatKeys, v)

data MExprMapX v
    = MM  {  mm_app   :: MExprMap (MExprMap v)
          ,  mm_fvar  :: Map Var v
          ,  mm_pvar  :: Maybe v     -- First occurrence of a pattern var
          ,  mm_xvar  :: PatOccs v   -- Subsequent occurrence of a pattern var
               -- SG: I propose to combine |mm_pvar, mm_xvar| and have a single
               -- |mm_pvar :: PatKeyMap v|. We can resolve first (flex) and second
               -- or subseq (rigid) occs as we go.
           }
    | EmptyMM
type PatOccs v = [(PatKey,v)]
\end{code}
The client-visible |MExprMap| with values of type |v|
is a matching trie |MExprMapX| with values of type |(PatKeys,v)|,
as described in \Cref{sec:patkeymap}.
The trie |MExprMapX| has four fields, one for each case in the pattern.
The first two fields deal with literals and applications, just as before. The third deals with the \emph{binding} occurrence
of a pattern variable $\pv{}$, and the fourth with a \emph{bound} occurrence of
a pattern variable $\pvo{i}$.

\sg{I think that most queries will have multiple Apps at the top-level and the
head will be a variable. It is a shame that we have to chase n pointers for an
n-ary application to get to the head! I think it would would be much more
efficient to store the list of App args in mm\_fvar, like}
\begin{spec}
         , mm_fvar :: Map Var (ListMap Expr v)
\end{spec}
\sg{(NB: PatVars can't occur in app heads that way. If we want them to, we have to give
mm\_pvar a similar treatment.) IIRC, that is what Lean's DiscTree does, and it's
also how we implement RULE matching (grep for ru\_fn)...}

The core lookup function looks like this:
%{
%if style == newcode
%format lkMExpr = "lkMExpr0"
%endif
\begin{code}
lkMExpr :: forall v. Expr -> (PatSubst, MExprMapX v) -> Bag (PatSubst, v)
\end{code}
As well as the target expression |Expr| and the trie, the lookup function also takes
a |PatSubst| that gives the bindings for pattern variable bound so far.
It returns a bag of results, since more than one entry in the trie may match,
each paired with the |PatSubst| that binds the pattern variables.
A |PatSubst| carries not only the current substition, but also (like a |BoundVarMap|, \Cref{sec:binders})
the next free pattern key:
\begin{code}
data PatSubst = PS { ps_next  :: PatKey, ps_subst :: Map PatKey Expr }
type PatKey = Int

emptyPatSubst :: PatSubst
emptyPatSubst = PS { ps_next = 0, ps_subst = Map.empty }

extendPatSubst :: Expr -> PatSubst -> PatSubst
extendPatSubst e (PS { ps_next = next, ps_subst = subst })
  = PS { ps_next = next + 1, ps_subst = Map.insert next e subst }

lookupPatSubst :: PatKey -> PatSubst -> Expr
lookupPatSubst pat_key (PS { ps_subst = subst })
  = case Map.lookup pat_key subst of
      Just expr -> expr
      Nothing   -> error "Unbound key"
\end{code}
%}

Given |lkMExpr| we can write |lookupMExpr|,
the externally-callable lookup function:
\begin{code}
lookupMExpr :: Expr -> MExprMap v -> Bag (Match v)
lookupMExpr e m = fmap rejig (lkMExpr e (emptyPatSubst, m))
  where
    rejig :: (PatSubst, (PatKeys, v)) -> Match v
    rejig (ps, (pkmk, v)) = (map (lookupPatKey ps) pkmk, v)

lookupPatKey :: PatSubst -> (PatVar,PatKey) -> (PatVar,Expr)
lookupPatKey subst (pat_var, pat_key) = (pat_var, lookupPatSubst pat_key subst)
\end{code}
Here |lookupMExpr| is just an impedence-matching shim around
a call to |lkMExpr| that does all the work.  Notice that the
input.  The latter returns a bag of |(PatSubst, (PatKeys, v))|
values, which the function |rejig| converts into the
the |Match v| results that we want.  The ``unbound key''
failure case in |lookupPatSubst| means that
|PatKeys| in a looked-up value asks for a key that is not
bound in the pattern.  The insertion function will ensure that this
never occurs.

Now we can return to the recursive function that does all the work: |lkMExpr|:
\begin{code}
lkMExpr :: forall v. Expr -> (PatSubst, MExprMapX v) -> Bag (PatSubst, v)
lkMExpr e (psubst, mt)
  = pat_var_bndr `Bag.union` pat_var_occs `Bag.union` look_at_e
  where
     pat_var_bndr :: Bag (PatSubst, v)
     pat_var_bndr = case mm_pvar mt of
                      Just v  -> Bag.single (extendPatSubst e psubst, v)
                      Nothing -> Bag.empty

     pat_var_occs :: Bag (PatSubst, v)
     pat_var_occs = Bag.fromList [ (psubst, v)
                                 | (pat_var, v) <- mm_xvar mt
                                 , e == lookupPatSubst pat_var psubst ]
                                 -- SG: the equality check here might get very costly, right?
                                 -- Although it probably only matters for pattern/pattern comparisons, e.g.,
                                 -- Are the following patterns compatible?
                                 --   ([x], f <huge> x .. x) and ([y], f y y .. y)
                                 -- A union/find would help here, I think, so that
                                 -- we record that x and y are equal (after we check
                                 -- the second arg pair) and don't have to compare
                                 -- the substitution of x and y, <huge>, multiple times.

     look_at_e :: Bag (PatSubst, v)
     look_at_e = case e of
        Var x     -> case Map.lookup x (mm_fvar mt) of
                        Just v  -> Bag.single (psubst,v)
                        Nothing -> Bag.empty
        App e1 e2 -> Bag.concatMap (lkT (D dbe t2)) $
                     lkT (D dbe t1) (tsubst, mem_fun mt)
\end{code}
The bag of results is the union of three possibilities, as follows. (Keep in mind that a |MExprMap| represents \emph{many} patterns simultaneously.)
\begin{itemize}
\item |pat_var_bndr|: we consult the |mm_pvar|, if it contains |Just v| then at least one of the patterns in this trie has a pattern binder $\pv{}$ at this spot.  In that case we can simply bind the next free pattern variable (|ps_next|) to |e|, and return a singleton bag.
\item |pat_var_occs|: any of the bound pattern varaibles might have an occurrence $\pvo{i}$ at this spot, and a list of such bindings is held in |pat_var_occs|.  For each, we must do an equality check between the target |e| and the expression bound to that pattern variable (found via |lookupPatSubst|).  We return a bag of all values for which the equality check succeeds.
  \item |look_at_e| corresponds exactly to the cases we saw before in \Cref{sec:Expr}.   The only subtlety is that we are are returning a \emph{bag} of results, but happily the Kleisli composition operator |(>=>)| (\Cref{fig:library}) works for any monad, including bags.
\end{itemize}

\subsection{Altering a matching trie}

\simon{Too much code, I know; but this section is one of the key contributions of the paper.}

How did the entries in our map get their |PatKeys|?  That
is, of course, the business of |insert|, or more generally
|alter|.  The key, recursive function must carry inwards a mapping
from pattern variables to pattern keys; we can simply re-use |BoundVarMap|
from \Cref{sec:binders} for this purpose.  The exact signature for the function takes
a bit of puzzling out, and is worth comparing with its predecessor in \Cref{sec:alter}:
\begin{code}
type PatKeyMap = BoundVarMap   -- We re-use BoundVarMap

xtMExpr :: Set PatVar -> Expr -> (PatKeyMap -> XT a)
         -> PatKeyMap -> MExprMapX v -> MExprMapX v
\end{code}
It is unsurprising the the function is given the set of pattern variables, so that it
can distinguish pattern variables from free variables.  It also takes a |PatKeyMap|, the
current binding of already-encountered pattern variables to their pattern keys;
when it completes the lookup it passes that completed binding map to the ``alter'' function.

Given this workhorse, we can build the client-visible |insert| function\footnote{|alter| is not much harder.}:
\begin{code}
insertExpr :: forall v. [Var]     -- Pattern variables
                         -> Expr  -- Pattern
                         -> v -> MExprMap v -> MExprMap v
insertExpr pat_vs e v mm
  = xtExpr (Set.fromList pat_vs) e xt emptyBVM mm
  where
    xt :: PatKeyMap -> XT (PatKeys, v)
    xt pkm _ = Just (map inst_key pat_vs, v)
     -- The @"_"@ means just overwrite previous value
     where
        inst_key :: PatVar -> (PatVar, PatKey)
        inst_key x = case lookupBVM x pkm of
                         Nothing -> error ("Unbound pattern variable " ++ x)
                         Just pk -> (x, pk)
\end{code}
This is the code that builds the |PatKeys| in the range of the map.
It does so using the |PatKeyMap| accumulated by |xtExpr| and
finally passed to the local function |xt|.

Now we can define the workhorse, |xtExpr|:
\begin{code}
xtExpr pvs e xt pkm mm
  = case e of
      App e1 e2 -> mm { mm_app = xtMExpr pvs e1 (liftXTS (xtMExpr pvs e2 xt))
                                         pkm (mm_app mm) }

      Var x | Just xv <- lookupBVM x pkm
             -> -- Second or subsequent occurrence of a pattern variable
                mm { mm_xvar = xtPatVarOcc xv (xt pkm) (mm_xvar mm) }

             | x `Set.member` pvs
             -> -- First occurrence of a pattern variable
                mm { mm_pvar = xt (extendBVM x pkm) (mm_pvar mm) }

             | otherwise
             -> -- A free variable
                mm { mm_fvar = Map.alter (xt pkm) x (mm_fvar mm) }

liftXTS :: (PatKeyMap -> MExprMap v -> MExprMap v)
        -> PatKeyMap -> Maybe (MExprMap v) -> Maybe (MExprMap v)
liftXTS xt pkeys Nothing  = Just (xt pkeys EmptyMM)
liftXTS xt pkeys (Just m) = Just (xt pkeys m)

xtPatVarOcc :: PatKey -> XT v -> PatOccs v -> PatOccs v
xtPatVarOcc key f []
  = xtCons key (f Nothing) []
xtPatVarOcc key f ((key1,x):prs)
  | key == key1 = xtCons key (f (Just x)) prs
  | otherwise   = (key1,x) : xtPatVarOcc key f prs

xtCons :: PatKey -> Maybe a -> PatOccs a -> PatOccs a
xtCons _   Nothing  pat_occs = pat_occs
xtCons key (Just x) pat_occs = (key,x) : pat_occs
\end{code}

\subsection{Most specific match}

\Cref{sec:matching-spec} described the goal of returning only the \emph{most specific matches} from
a lookup.  In GHC today, the lookup returns \emph{all} matches, and these matches are then
exhaustively compared against each other; if one is more specific than (a substitution instance of) another, the latter is discarded.

A happy consequence of the trie representation is that a one-line change suffices
to return only the most-specific matches.  We simply modify the definition |lkMExpr| from \Cref{sec:matching-lookup} as follows:
%{
%if style == poly
%format as_before = "\ldots \text{as before} \ldots"
%else
%format as_before = "pat_var_occs :: Bag (PatSubst, v); pat_var_occs = undefined; look_at_e :: Bag (PatSubst, v); look_at_e = undefined; pat_var_bndr :: Bag (PatSubst, v); pat_var_bndr = undefined"
%format lkMExpr = "lkMExpr2"
%endif
\begin{code}
lkMExpr e (psubst, mt)
  | Bag.null pat_var_occs && Bag.null look_at_e
  = pat_var_bndr
  | otherwise
  = pat_var_occs `Bag.union` look_at_e
  where
    as_before
\end{code}
%}
That is, we only return the matches obtained by matching a pattern variable (|pat_var_bndr|) if there
are no more-specific matches (namely |pat_var_occs| and |look_at_e|).  It is a happy
consequence of the trie structure that this simple (and efficient in execution terms) change suffices
to return the most-specific matches.

\sg{But that notion of most-specific is biased towards specificity happening
early in the App chain, if I'm not mistaken. So given the map $\{(([x],
f~x~True), 1), (([y], f~True~y), 2)\}$, the most-specific match of $f~True~True$
will be $2$: the second pattern is more specific in the first App arg, while
the first one has simply an unbound patvar in that position. But actually I'd
consider $1$ just as specific, albeit incomparable to $2$. In effect, you're
forcing a lexicographic ordering on patterns where I don't think there should
be one.}

\subsection{Unification}

\section{Evaluation} \label{sec:eval}

So far, we have seen that trie maps offer at least one significant advantage
over other kinds of maps like ordered maps or hash maps: The ability to do a
matching lookup in \Cref{sec:matching}. In this section, we will see that
query performance is another advantage. Our implementation of trie maps in
Haskell can generally compete in performance with other map data structures,
while significantly outperforming traditional map implementations on some
operations.

\subsection{Runtime}

\begin{table}

  \caption{Benchmarks of different operations over our trie map |ExprMap| (TM),
  ordered maps |Map Expr| (OM) and hash maps |HashMap Expr| (HM), varying the
  size parameter $N$.
  We give the speedup of OM and HM relative to absolute runtime measurements for
  TM. Digits whose order of magnitude is no larger than that of twice the standard
  deviation are marked by squiggly lines.}
  \begin{tabular}{l rrr rrr rrr}
  \toprule
  $N$  & \multicolumn{3}{c}{\textbf{10}} & \multicolumn{3}{c}{\textbf{100}} & \multicolumn{3}{c}{\textbf{1000}} \\
       \cmidrule(lr{.5em}){2-4} \cmidrule(lr{.5em}){5-7} \cmidrule(lr{.5em}){8-10}
  Data structure  & \multicolumn{1}{c}{TM} & \multicolumn{1}{c}{OM} & \multicolumn{1}{c}{HM}
       & \multicolumn{1}{c}{TM} & \multicolumn{1}{c}{OM} & \multicolumn{1}{c}{HM}
       & \multicolumn{1}{c}{TM} & \multicolumn{1}{c}{OM} & \multicolumn{1}{c}{HM} \\
  \midrule
  \input{bench-overview.tex-incl}
  \bottomrule
  \end{tabular}

  \label{fig:runtime}
\end{table}

\Cref{fig:runtime} presents the results of measuring runtime on
\sg{insert architecture when finalising benchmarks}. All runtime measurements
were conducted as microbenchmarks using the \texttt{criterion}%
\footnote{\url{https://hackage.haskell.org/package/criterion}}
benchmark suite.

\subsubsection*{Setup}
All benchmarks except the \benchname{fromList*} variants are handed a pre-built
map containing $N$ expressions, each consisting of roughly $N$ |Expr| data
constructors and drawn from a pseudo-random source with a fixed (and thus
deterministic) seed. $N$ is varied between 10 and 1000.
Benchmarks ending in \benchname{\_lam}, \benchname{\_app1}, \benchname{\_app2}
add a shared prefix to each of the $N$ expressions before building the initial
map:
\begin{itemize}
  \item \benchname{\_lam} wraps $|(Lam "$")|^N$ around each expression
  \item \benchname{\_app1} wraps $|(Lit "$" `App`)|^N$ around each expression
  \item \benchname{\_app2} wraps $|(`App` Lit "$")|^N$ around each expression
\end{itemize}
Where |"$"| is a name that doesn't otherwise occur in the generated expressions.

\begin{itemize}
  \item The \benchname{lookup\_all*} family of benchmarks looks up every
        expression that is part of the map. So for a map of size 100, we will
        perform 100 lookups of expressions each of which have approximately size
        100. \benchname{lookup\_one} will look up just one expression that is
        part of the map.
  \item \benchname{insert\_lookup\_one} will insert a random expression into the
        initial map and immediately look it up afterwards. The lookup is to
        ensure that any work delayed by laziness is indeed forced.
  \item The \benchname{fromList*} family benchmarks a naïve |fromList|
        implementation on |ExprMap| against the tuned |fromList| implementations
        of the other maps, measuring map creation performance from batches.
  \item \benchname{fold} simply sums up all values that are stored in the map
        (which stores |Int|s).
\end{itemize}

\subsubsection*{Querying}
The results show that lookup in |ExprMap| often wins against |Map Expr| and
|HashMap Expr|. The margin is small on the completely random |Expr|s of
\benchname{lookup\_all}, but realistic applications of |ExprMap| often store
|Expr|s with some kind of shared structure. The \benchname{\_lam} and
\benchname{\_app1} variants show that |ExprMap| can win big time against
an ordered map representation: |ExprMap| looks at the shared prefix exactly
once one lookup, while |Map| has to traverse the shared prefix on each of its
$\mathcal{O}(\log n)$ comparisons. As a result, the gap between |ExprMap| and
|Map| widens as $N$ increases, confirming an asymptotic difference.
The advantage is less pronounced in the \benchname{\_app2} variant, presumably
because |ExprMap| can't share the common prefix here: It turns into an
unsharable suffix in the pre-order serialisation, blowing up the trie map
representation compared to its sibling \benchname{\_app1}.

Although |HashMap| loses on most benchmarks compared to |ExprMap| and |Map|, it
performs much more consistently than |Map|. We believe it that is due to the
fact that it is enough to traverse the |Expr| once to compute the hash, thus
it is expected to scale similarly as |ExprMap|.

Comparing the \benchname{lookup\_all*} measurements of the same map data
structure on different size parameters $N$ reveals a roughly cubic correlation.
That seems plausible given that $N$ linearly affects map size, expression size
and number of lookups. But realistic workloads tend to have much larger map
sizes than expression sizes!

\begin{table}
  \caption{Varying expression size $E$ and map size $M$ independently on benchmark
  \benchname{lookup\_all}.}
  \begin{tabular}{r rrr rrr rrr}
  \toprule
  \multirow{2}{*}{\diagbox{$E$}{$M$}} & \multicolumn{3}{c}{\textbf{10}}
                                      & \multicolumn{3}{c}{\textbf{100}}
                                      & \multicolumn{3}{c}{\textbf{1000}} \\
  \cmidrule(lr{.5em}){2-4} \cmidrule(lr{.5em}){5-7} \cmidrule(lr{.5em}){8-10}
                     & TM & OM & HM
                     & TM & OM & HM
                     & TM & OM & HM \\
  \midrule
  \input{bench-lookup.tex-incl}
  \bottomrule
  \end{tabular}

  \label{fig:runtime-lookup}
\end{table}

Focusing on \benchname{lookup\_all}, we measured performance when independently
varying map size $M$ and expression size $E$. The results in
\Cref{fig:runtime-lookup} show that |ExprMap| scales even better than |Map| when
we increase $M$ and leave $E$ constant. The time measurements for |ExprMap|
appear to grow linearly with $M$. Considering that the number of lookups also
increases $M$-fold, it seems the cost of a single lookup remained constant,
despite the fact that we store varying numbers of expressions in the trie map.
By contrast, fixing $M$ but increasing $E$ makes |Map| easily catch up on lookup
performance with |ExprMap|, ultimately outpacing it. |HashMap| shows performance
consistent with |ExprMap| but is a bit slower, as before.

\subsubsection*{Building}

\begin{itemize}
  \item Quite fast, but will be outpaced by |Map| for huge $E$
  \item |fromList| is generally slow. No ideas for improvement
  \item Likewise |fold|. Don't do it
\end{itemize}

\subsection{Space}

Another table here

\section{Related work} \label{sec:related}

\subsection{Matching triemaps in automated reasoning}

Matching triemaps, also called \emph{term indexing}, have been used in the automated
reasoning community for decades.
An automated reasoning system has
hundreds or thousands of axioms, each of which is quantified over
some variables (just like the RULEs described in \Cref{sec:matching-intro}). Each of these
axioms might apply at any sub-tree of the term under consideration, soa efficient
matching of many axioms is absolutely central to the performance of these systems.

This led to a great deal of work in so-called \emph{discrimination trees}, starting
in the late 1980's, which is beautifully surveyed in the Handbook of Automated Reasoning
\cite[Chapter 26]{handbook:2001}.
Automated theorem provers such as Vampire, E, Z3, and (to a lesser extent) interactive
theorem provers like Coq, Isabelle, and Lean, make extensive use of discrimination trees.

Discrimination trees save work when keys have a common \emph{prefix}.
\emph{Substitution trees} (also surveyed in the same Handbook) go
further and try to save work when keys have common \emph{sub-trees};
but the extra complexity does not seem to pay its way in practice, and
substitution trees do not appear to be used in production applications.
\simon{Leonardo, is that true; I think it's what you said.}
\sg{Is that so? I see it is used in Z3:
\url{https://github.com/Z3Prover/z3/blob/master/src/ast/substitution/substitution_tree.h}
But it only has a single use site, in
\url{https://github.com/Z3Prover/z3/blob/21e59f7c6e5033006265fc6bc16e2c9f023db0e8/src/muz/transforms/dl_mk_rule_inliner.h}.}

Seen from a sufficient distance, our work is very close to discrimination trees -- we
have simply re-presented discrimination trees in Haskell.
But doing so is insightful, and there are also numerous differences of detail:
\begin{itemize}
\item We present our triemaps as a library written in a statically typed functional
  language, whereas the discrimination tree literature tends to assume an implementation in C,
  and gives algorithms in pseudocode.

\item Our triemaps provide a full range of operations, including alter, union, and fold,
  wheres the automated-reasoning applications focus almost exclusively on insert and lookup.

\item We build triemaps for many
different data types, whereas the discrimination tree literature tends to assume
a single built-in data type of terms.

\item We use type classes and polymorphism to make it easy to build triemaps
  over polymorphic types like lists (\Cref{sec:class}).
\end{itemize}

\sg{Maybe talk about section 14 of the handbook, where different ways (such as delayed
equality constraints) are proposed to deal with non-linearity instead of
numbering all pattern variable occurrences, which loses sharing.}

\subsection{Haskell triemaps}

\subsection{Notes from Sebastian}

\begin{itemize}
\item Using a FSM; e.g \emph{Interpreted Pattern Match Execution} by Jeff Niu, a UW undergrad intern at Google.  https://docs.google.com/presentation/d/1e8MlXOBgO04kdoBoKTErvaPLY74vUaVoEMINm8NYDds/edit?usp=sharing

\item Matching multiple strings.
\end{itemize}
There is rich literature on radix trees, which incorporate the Singleton optimisation simply as ``each node that is the only child is merged with its parent'', and an abundance of related work in the theorem proving community available under the term ``Discrimination Tree'' and ``Term Indexing''. I think it would help the paper if instead of starting from ``an API for finite maps''/tries as a baseline, it would start from "an API for term indexing"/discrimination trees as decribed in the Handbook of Automated Reasoning (2001), for example. I'll have access to a hard copy in a couple of days and can then report on the contents...

Here's a GH issue that suggests using Discrimination Trees to speed up Hoogle queries: https://github.com/ndmitchell/hoogle/issues/250. That thread generally seems like a good source of references to consider. It suggests that discrimination trees are but the simplest data structure to perform term indexing.
Remy Goldschmidt (@@taktoa, the GH issue creator) even provides a model implementation of discrimination trees in Haskell: https://gist.github.com/taktoa/7a4d77ebc3a312dd69bb19199d30863b

Here's a paper from 1994 claiming to be faster than discrimination trees: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.43.823 (In case you're wondering, I'm not affiliated with the author at all.)

It appears that at least since 2009, the 1994 approach has already been extended to a higher-order pattern scenario (allowing e.g. miller pattern unification): https://dl.acm.org/doi/10.1145/1614431.1614437

\bibliography{triemap}

\end{document}
