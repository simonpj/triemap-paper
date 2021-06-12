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

\newcommand{\simon}[1]{{\bf SLPJ}: {\color{dkcyan} #1} {\bf End SLPJ}}
\newcommand{\js}[1]{{\bf JS}: {\color{olive} #1} {\bf End JS}}
\newcommand{\rae}[1]{{\bf RAE}: {\color{dkblue} #1} {\bf End RAE}}
\newcommand{\sg}[1]{{\bf SG}: {\color{darkbrown} #1} {\bf End SG}}

\newcommand{\bv}[1]{\#_{#1}}    -- Lambda-bound variable occurrence
\newcommand{\pv}[1]{\$_{#1}}    -- Pattern variable binder
\newcommand{\pvo}[1]{\%_{#1}}   -- Pattern variable occurrence


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

\author{Josef Sveningsson}

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
from keys to values, in which the key type is a tree}.
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
for this paper. For the sake of concreteness,
you may wish to imagine it is simply a string:
\begin{code}
type Var = String
\end{code}
%}
The data type |Expr| is capable of representing expressions like |add 2 3| and
|\lambda x. add x 3|. \rae{Well, not really, because it doesn't support literals.
Is this worth fixing/rephrasing?} We will use this data type throughout the paper, because it
has all the features that occur in real expression data types: free variables like |add|,
represented by a |Var| node;
lambdas which can bind variables (|Lam|), and occurrences of those bound variables (|Var|);
and nodes with multiple children (|App|).  A real-world expression type would have
many more constructors.

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
key is a type.

\subsection{The interface of of a finite map}

What API might such a map have? Building on the design of widely
used functions in Haskell (see \cref{fig:containers}), we
seek these basic operations:
\begin{code}
empty   :: ExprMap v
lookup  :: Expr -> ExprMap v -> Maybe v
alter   :: Expr -> XT v -> ExprMap v -> ExprMap v
\end{code}
The functions |empty| and |lookup| should be
self-explanatory.  The function |alterTM| is a standard
generalisation of |insert|: instead of providing just
a new element to be inserted, the caller provides a
\emph{transformation} |XT v|, an
abbreviation for |Maybe v -> Maybe v|.  This function
transforms the existing value associated with key, if any (hence the
input |Maybe|), to a new value, if any (hence the output |Maybe|). 
These fundamental operations on a finite map must obey these properties:
\begin{code}
property propLookupEmpty (e)                       (lookup e empty             ^^^^)  (Nothing)
property propLookupAlter (e m xt)                  (lookup e (alter e xt m)    ^^^^)  (xt (lookup e m))
propertyImpl propWrongElt (e1 e2 m xt) (e1 /= e2)  (lookup e1 (alter e2 xt m)  ^^^^)  (lookup e1 m)
\end{code}

We can easily define |insert| and |delete| from |alter|:
\begin{code}
insert :: Expr -> v -> ExprMap v -> ExprMap v
insert e v = alter e (\_ -> Just v)

delete :: Expr -> ExprMap v -> ExprMap v
delete e = alter e (\_ -> Nothing)
\end{code}
You might wonder if, for the purposes of this paper we could just define |insert|,
leaving |alter| for the Appendix, but as we will see in \Cref{sec:alter}, our
approach using tries fundamentally requires the generality of |alter|.

We would also like to support other standard operations on finite maps, including
\begin{itemize}
\item An efficient union operation to combine two finite maps into one:
\begin{code}
union :: ExprMap v -> ExprMap v -> ExprMap v
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

\subsection{Non-solutions} \label{sec:ord}

At first sight, our task can be done easily: define a total order on
|Expr|
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

\subsection{Extra feature: Alpha-renaming}

Recall that the type |Expr| is the \emph{key} of our |ExprMap| type.
We do not want our programming language to discern between the expressions
|\x -> x| and |\y -> y|, and so we would expect insertion and lookup to be insensitive to
$\alpha$-renaming. That is, the correctness properties we list above
must use $\alpha$-equivalence when comparing expressions for equality.

The naive approach can, with a small amount of effort, accommodate $\alpha$-equivalence.
We can convert our key expression to one using de Bruijn levels or indices,
and then work with that. However, doing so requires making a copy of the input
key, a potentially expensive extra step.
\rae{Remind me why hashing-modulo-alpha is hard, referring to
\cite{alpha-hashing}.}

\subsection{Extra feature: Matching} \label{sec:matching-intro}

Beyond just the basic finite maps we have described, our practical setting
in GHC demands more: we want to a lookup that does \emph{matching}.  GHC supports
so-called \emph{rewrite rules} \cite{rewrite-rule-paper}, which the user can specify like this:
\begin{code}
prag_begin RULES "map/map" forall f g xs. map f (map g xs) = map (f . g) xs prag_end
\end{code}
This rule asks the compiler to rewrite any target expression that matches the shape
of the left-hand side (LHS) of the rule into the right-hand side
(RHS).  We often use the term \emph{pattern} to describe the LHS.
The pattern is explicitly quantified over the \emph{pattern variables}
(here |f|, |g|, and |xs|) that
can be bound during the matching process.  In other words, \emph{we seek a substitution
for the pattern variables that makes the pattern equal to the target expression}.
For example, if the program we are compiling contains the expression
|map double (map square nums)|, we would like to produce a substitution
|f ||-> double, g ||-> square, xs ||-> nums| so that the substituted RHS
becomes |map (double . square) nums|; we would replace the former expression
with the latter in the code under consideration.

Now imagine that we have thousands of such rules.  Given a target
expression, we want to consult the rule database to see if any
rule matches.  One approach would be to look at the rules one at a
time, checking for a match, but that would be slow if there are many rules.
How could we do this more efficiently?

Of course, the pattern might itself have bound variables, and we would
like to be insensitive to alpha-conversion for those. For example:
\begin{code}
prag_begin RULES "map/id"  map (\x -> x) = \x -> x prag_end
\end{code}
We want to find a successful match if we see a call |map (\y -> y)|,
even though the bound variable has a different name.

This more flexible lookup---allowing matching as we go---is a valuable ability.
For example, GHC's lookup for
type-class instances and for type-family instances can again have thousands
of candidates. We would like to find a matching candidate more efficiently
than by linear search.

Support for pattern variables is hard to reconcile with many standard approaches to
implementing finite maps.  For example, representing the finite map as a binary tree,
and performing comparisons at the nodes to determine which sub-tree holds the key,
seems an approach that is hard or impossible to extend to support matching.

We thus now present our data structure that easily accommodates all of our
desiderata: the TrieMap.

\section{Tries} \label{sec:ExprS}

A rather standard approach to a finite map in which the key has internal structure
is to use a \emph{trie} \cite{trie}.  Let us consider a simplified
form of expression:
\begin{code}
data ExprS = VarS Var | AppS ExprS ExprS
\end{code}
We leave lambdas out for now,
so that all |Var| nodes represent free variables, which are constants.
We will return to lambdas in \Cref{sec:binders}.

\subsection{The basic idea} \label{sec:basic} \label{sec:alter}

Here is a trie-based implemenation for |ExprS|:
%{
%if style == newcode
%format ExprSMap = "ExprSMap0"
%format ESM = "ESM0"
%format esm_var = "esm_var0"
%format esm_app = "esm_app0"
%format lookupExprS = "lookupExprS0"
%format alterExprS = "alterExprS0"
%format liftXT = "liftXT0"
%format emptyExprS = "emptyExprS0"
%endif
\begin{code}
data ExprSMap v = ESM  { esm_var  :: Map Var v
                       , esm_app  :: ExprSMap (ExprSMap v) }
\end{code}
Here |Map Var v| is any standard, existing finite map keyed by |Var|.

One way to understand this slightly odd data type is to study its lookup function:
\begin{code}
lookupExprS :: ExprS -> ExprSMap v -> Maybe v
lookupExprS e (ESM { esm_var = m_var, esm_app = m_app })
  =  case e of
      VarS x      -> Map.lookup x m_var
      AppS e1 e2  ->  case lookupExprS e1 m_app of
                        Nothing  -> Nothing
                        Just m1  -> lookupExprS e2 m1
\end{code}
This function pattern-matches on the target |e|.  The |VarS| alternative
says that to look up a variable occurrence, just look that variable up in the
|esm_var| field.
But if the expression is an |AppS e1 e2| node, we first look up |e1|
in the |esm_app| field, \emph{which returns a finite map}.  We then look up |e2|
in that map.  Each distinct |e1| yields a different map in which to look up |e2|.

We can substantially abbreviate this code, at the expense of making it more cryptic, thus:
%{
%if style == newcode
%format lookupExprS = "lookupExprS2"
\begin{code}
lookupExprS :: ExprS -> ExprSMap v -> Maybe v
  -- we need this type signature because the body is polymorphic recursive
\end{code}
%endif
\begin{code}
lookupExprS (VarS x)      = esm_var  >.> Map.lookup x
lookupExprS (AppS e1 e2)  = esm_app  >.> lookupExprS e1 >=> lookupExprS e2
\end{code}
%}
We use some simple composition combinators, whose types are given in \Cref{fig:library}
to chain together the component pieces.  The function |esm_var :: ExprSMap v -> Map Var v|
is the auto-generated selector that picks |esm_var| field from an |ESM| record.
The functions |>.>| and |>=>| are forward composition
operators, respectively monadic and non-monadic, that chain the individual operations together.
Finally, we have $\eta$-reduced the definition, by omitting the |m| parameter.
These abbreviations become quite worthwhile when we add more constructors to the key data type.

Notice that in contrast to the approach of \Cref{sec:ord}, \emph{we never compare two expressions
for equality or ordering}.  We simply walk down the |ExprMap| structure, guided
at each step by the next node in the target.  (We typically use the term ``target'' for the
key we are looking up in the finite map.)

This definition is extremely short and natural. But it conceals a hidden
complexity: \emph{it requires polymorphic recursion}. The recursive call to |lookupExprS e1|
instantiates |v| to a different type than the parent function definition.
Haskell supports polymorphic recurision readily, provided you give type signature to
|lookupExprS|, but not all languages do.

\subsection{Modifying tries} \label{sec:alter}

It is not enough to look up in a trie -- we need to \emph{build} them too!
Here is the code for |alter|:
\begin{code}
alterExprS :: ExprS -> XT v -> ExprSMap v -> ExprSMap v
alterExprS e xt m@(ESM { esm_var = m_var, esm_app = m_app })
  =  case e of
       VarS x      -> m { esm_var  = Map.alter xt x m_var }
       AppS e1 e2  -> m { esm_app  = alterExprS e1 (liftXT (alterExprS e2 xt)) m_app }

liftXT :: (ExprSMap v -> ExprSMap v) -> XT (ExprSMap v)
liftXT f Nothing    = Just (f emptyExprS)
liftXT f (Just m)   = Just (f m)
\end{code}
%}
In the |VarS| case, we must just update the map stored in the |esm_var| field,
using the |Map.alter| function from \Cref{fig:containers};
in Haskell the notation ``|m { fld = e }|'' means the result
of updating the |fld| field of record |m| with new value |e|.
In the |AppS| case we look up |e1| in |m_app|;
we should find a |ExprSMap| there, which we want to alter with |xt|.
We can do that with a recursive call to |alterExprS|, using |liftXT|
for impedence-matching.

The |AppS| case shows why we need the generality of |alter|.
Suppose we attempted to define an apparently-simpler |insert| operations.
Its equation for |(AppS e1 e2)| would look up |e1| --- and would then
need to \emph{alter} that entry (an |ExprSMap|, remember) with the result of
inserting |(e2,v)|.  So we are forced to define |alter| anyway.

We can abbreviate the code for |alterExprS| using combinators, as we did in the case of
lookup, and doing so pays dividends when the key is a data type with
many constructors, each with many fields.  However, the details are
fiddly and not illuminating, so we omit them here.  Indeed, for the
same reason, in the rest of this paper we will typically omit the code
for |alter|, though the full code is available in the
Appendix.

\subsection{Unions of maps}

A common operation on finite maps is to take their union:
\begin{code}
unionExprS :: ExprSMap v -> ExprSMap v -> ExprSMap v
\end{code}
In tree-based implementations of finite maps, such union operations can be tricky.
The two trees, which have been built independently, might not have the same
left-subtree/right-subtree structure, so some careful re-alignment may be requried.
But for tries there are no such worries --
their structure is identical, and we can simply zip them together.  There is one
wrinkle: just as we had to generalise |insert| to |alter|,
to accomodate the nested map in |esm_app|, so we need to generalise |union| to |unionWith|:
\begin{code}
unionWithExprS :: (v -> v -> v) -> ExprSMap v -> ExprSMap v -> ExprSMap v
\end{code}
When a key appears on both maps, the combining function is used to
combine the two corresponding values.
With that generalisation, the code is as follows:
\begin{code}
unionWithExprS f (ESM { esm_var = m1_var, esm_app = m1_app })
                 (ESM { esm_var = m2_var, esm_app = m2_app })
  = ESM { esm_var = Map.unionWith f m1_var m2_var
        , esm_app = unionWithExprS (unionWithExprS f) m1_app m2_app }
\end{code}
It could hardly be simpler. \rae{What about empty/single maps? They're missing here.}

\subsection{Folds and the empty map} \label{sec:fold}

Of course, we need an empty trie. Here is one way to define such a thing:
%{
%if style == newcode
%format emptyExprS = "emptyExprS0"
%format foldrExprS = "foldrExprS0"
%format sizeExprS = "sizeExprS0"
%format ExprSMap = "ExprSMap0"
%format esm_var = "esm_var0"
%format esm_app = "esm_app0"
%format ESM = "ESM0"
%endif
\begin{code}
emptyExprS :: ExprSMap v
emptyExprS = ESM { esm_var = Map.empty, esm_app = emptyExprS }
\end{code}
It is interesting to note that |emptyExprS| is an infinite, recursive structure:
the |esm_app| field refers back to |emptyExprS|.
This slightly strange definition works fine for lookup and alteration, but it fails
fundamentally when we want to iterate over the elements of the trie.

For example, suppose we wanted to count the number of elements in the finite map; in |containers|
this is the function |Map.size| (\Cref{fig:containers}).  We might try
%{
%if style == poly
%format undefined = "???"
%endif
\begin{code}
sizeExprS :: ExprSMap v -> Int
sizeExprS (ESM { esm_var = m_var, esm_app = m_app })
  = Map.size m_var + undefined
\end{code}
%}
We seem stuck because the size of the |m_app| map is not what we want: rather,
we want to add up the sizes of its elements, and we don't have a way to do that yet.
The right thing to do is to generalise to a fold:
\begin{code}
foldrExprS :: (v -> r -> r) -> r -> ExprSMap v -> r
foldrExprS k z (ESM { esm_var = m_var, esm_app = m_app })
  = Map.foldr k z1 m_var
  where
    z1 = foldrExprS kapp z m_app
    kapp m1 r = foldrExprS k r m1
\end{code}
%}
Here, in the binding for |z1| we fold over |m_app :: ExprSMap (ExprSMap v)|.
The function |kapp| is combines the map we find with the accumulator, by again
folding over the map with |foldrExprS|.

But alas, |foldrExprS| will never terminate!  It always invokes itself immediately
(in |z1|) on |m_app|; but that invocation will again recursively invoke
|foldrExprS|; and so on for ever.
The solution is simple: we just need an explicit representation of the empty map.
Here is one way to do it (we will see another in \Cref{sec:generalised}):
%{
%if style == newcode
%format ExprSMap = "ExprSMap1"
%format EmptyESM = "EmptyESM1"
%format ESM = "ESM1"
%format esm_var = "esm_var1"
%format esm_app = "esm_app1"
%endif
\begin{code}
data ExprSMap v = EmptyESM
                | ESM { esm_var :: Map Var v, esm_app :: ExprSMap (ExprSMap v) }

emptyExprS :: ExprSMap v
emptyExprS = EmptyESM

foldrExprS :: (v -> r -> r) -> r -> ExprSMap v -> r
foldrExprS k z EmptyESM                                   = z
foldrExprS k z (ESM { esm_var = m_var, esm_app = m_app }) = Map.foldr k z1 m_var
  where
    z1 = foldrExprS kapp z m_app
    kapp m1 r = foldrExprS k r m1
\end{code}
Equipped with a fold, we can easily define the size function, and another
that returns the range of the map:
\begin{code}
sizeExprS :: ExprSMap v -> Int
sizeExprS = foldrExprS (\_ n -> n+1) 0

elemsExprS :: ExprSMap v -> [v]
elemsExprS = foldrExprS (:) []
\end{code}
%}
\subsection{Singleton maps} \label{sec:singleton}

Suppose we start with an empty map, and insert a value
with a key (an |Expr|) that is large, say
\begin{spec}
  App (App (Var "f") (Var "x")) (Var "y")
\end{spec}
Looking at the code
for |alterExprS| in \Cref{sec:alter}, you can see that
because there is an |App| at the root, we will build an
|ESM| record with an empty |esm_var|, and an
|esm_app| field that is... another |ESM|
record.  Again the |esm_var| field will contain an
empty map, while the |esm_app| field is a further |ESM| record.

In effect, the key is linearised into a chain of |ESM| records.
This is great when there are a lot of keys with shared structure, but
once we are in a sub-tree that represents a single key-value pair it is
a rather inefficient way to represent the key.  So a simple idea is this:
when a |ExprMap| represents a single key-value pair, represent it
as directly a key-value pair!  Like this:
\begin{code}
data ExprSMap v = EmptyESM
                | SingleESM ExprS v
                | ESM { esm_var :: Map Var v, esm_app :: ExprSMap (ExprSMap v) }
\end{code}
The code for lookup practically writes itself:
%{
%if style = poly
%format dots = "\ldots"
%else
%format dots = undefined
%endif
\begin{code}
lookupExprS :: ExprS -> ExprSMap v -> Maybe v
lookupExprS e EmptyESM
  = Nothing
lookupExprS e1 (SingleESM e2 v2)
  = if e1 == e2 then Just v2
              else Nothing
lookupExprS e (ESM { esm_var = m_var, esm_app = m_app })
  = dots     -- Exactly as before
\end{code}
%}
Notice that in the |SingleESM| case we need equality on |Expr|,
to tell if the key being looked up, |k1| is the same as the key in
the |SingleESM|, namely |k2|.

The code for alter is more interesting, becuase it governs the shift from
|EmptyESM| to |SingleESM| and thence for |ESM|:
\begin{code}
alterExprS  :: ExprS -> XT v -> ExprSMap v -> ExprSMap v
alterExprS e xt EmptyESM
  = case xt Nothing of
      Nothing -> EmptyESM
      Just v  -> SingleESM e v

alterExprS e xt m@(SingleESM key v1)
  | e == key
  = case xt (Just v1) of
      Nothing -> EmptyESM
      Just v2  -> SingleESM e v2
  | otherwise
  = case xt Nothing of
      Nothing -> m
      Just v2 -> alterExprS key (\_ -> Just v1) $
                 alterExprS e   (\_ -> Just v2) $
                 ESM { esm_var = Map.empty, esm_app = EmptyESM }

alterExprS e xt m@(ESM { esm_var = m_var, esm_app = m_app })
  = case e of
      VarS x     -> m { esm_var = Map.alter xt x m_var }
      AppS e1 e2 -> m { esm_app = alterExprS e1 (liftXT (alterExprS e2 xt)) m_app }

liftXT :: (ExprSMap v -> ExprSMap v) -> XT (ExprSMap v)
liftXT f Nothing  = Just (f EmptyESM)
liftXT f (Just m) = Just (f m)
\end{code}
\sg{liftXT expands Nothing to emptyExprS but doesn't contract an emptyExprS result
from (f m) back to Nothing. So inserting and deleting a key will leave behind
the spine of an empty map for that key.}
Although we began by speaking of a map containing only one key-value pair,
this representation uses |ESM| while there are keys that share structure,
but as soon as we get into a sub-treee where there is no overlap, we revert
to |SingleESM|.

This optimisation makes a big difference in practice: see \Cref{sec:results}.

\subsection{Generalised singleton and empty maps} \label{sec:generalised}

Rather than implement the code for singleton maps and empty maps in every trie,
we can do it once and for all, like this:
\begin{code}
data SEMap m k v  -- Wrapper for singleton and empty map
  = EmptyTM | SingleTM k v | MultiTM (m v)

emptySEMap :: SEMap m k v
emptySEMap = EmptyTM

lkSEMap :: Eq k => (k -> m v -> Maybe v) -> k -> SEMap m k v -> Maybe v
lkSEMap _  _  EmptyTM                    = Nothing
lkSEMap _  tk (SingleTM k v) | tk == k   = Just v
                             | otherwise = Nothing
lkSEMap lk tk (MultiTM m)                = lk tk m
\end{code}
Here |lkSEMap| is responsible for the empty and singleton
cases, and delegates to the arugment function |lk| in all other cases.
Now we can return to the simpler code in \Cref{sec:basic}, and define
\begin{code}
type GExprSMap v = SEMap ExprSMap ExprS v

lookupExprSMap :: ExprS -> GExprSMap v -> Maybe v
lookupExprSMap = lkSEMap lookupExprS
\end{code}
The code for |xtSEMap|, and |alterExprSMap|, follows straightforwardly.

\subsection{Maps of higher kinds}

Suppose our expresions had multi-argument apply nodes, thus
%{
%if style == poly
%format dots = "\ldots"
%else
%format dots = "Ctor"
%format Expr = "Expr1"
%endif
\begin{code}
data Expr = dots
          | AppV Expr [Expr]
\end{code}
%}
Then we would need to built a trie keyed by a \emph{list} of |Expr|.
Since a list is just another algebraic data type, build with nil and cons,
we can use exactly the same approach, thus
\begin{code}
lookupListExpr :: [Expr] -> ListExprMap v -> Maybe v
\end{code}
But rather than build an implementation
for |[Expr]|, and then another for |[Decl]|, etc, we obviously
want to build a trie for lists of \emph{anything}, something like this \cite{hinze}:
%{
%if style == newcode
%format lookupList = "lookupList0"
%endif
\begin{code}
-- lookupList :: [k] -> ListMap k v -> Maybe v  \rae{The kinds don't work out there. What do we mean?}
\end{code}
%}
But this obviously cannot work: we need some type-class constraint on the key |k|,
saying that it can be used as the key of a trie.   That suggests
%{
%if style == newcode
%format lookupList = "lookupList1"
%endif
\begin{code}
lookupList :: TrieKey k => [k] -> TrieMap [k] v -> Maybe v

class Eq k => TrieKey k where
  type TrieMap k :: Type -> Type
  emptyTM  :: TrieMap k v    -- \rae{This type is ambiguous.}
  lookupTM :: k -> TrieMap k v -> Maybe v
  alterTM  :: k -> XT v -> TrieMap k v -> TrieMap k v
  foldTM   :: (v -> r -> r) -> r -> TrieMap k v -> r
\end{code}
%}
The class constraint |TrieKey k| says that the type |k|
can be used as the key of a triemap.
The class has an \emph{associated type}, |TrieMap k|,
a type-level function that transforms the type of the key into
the type of a trie for that key.  Now we can witness the fact that |ExprS| can be
used as the key of a triemap, like this:
%{
%if style == poly
%format dots = "\ldots"
%else
%format dots = "alterTM = undefined; foldTM = undefined"
%endif
\rae{Are there definitions for |alterTM| and |foldTM| for this, if we wanted them?}
\begin{code}
instance TrieKey ExprS where
  type TrieMap ExprS = SEMap ExprSMap ExprS
  emptyTM  = emptySEMap
  lookupTM = lkSEMap lookupExprS
  dots
\end{code}

All this puts us in a position to write the instance for lists:
\begin{code}
instance TrieKey k => TrieKey [k] where
  type TrieMap [k] = SEMap (ListMap (TrieMap k)) [k]
  emptyTM  = emptySEMap
  lookupTM = undefined -- lkSEMap lookupList  \rae{There is a type error here. Fix.}
  dots

data ListMap elt_m v = LM { lm_nil  :: Maybe v, lm_cons :: elt_m (ListMap elt_m  v) }

lookupList :: TrieKey k => [k] -> TrieMap [k] v -> Maybe v
lookupList []     = undefined  -- @lm_nil@  \rae{another type error}
lookupList (x:xs) = undefined  -- @lm_cons |> lookupTM x >=> lookupList xs@  \rae{another type error}
\end{code}
%}
The code for |alter| and |fold| is routine.

\section{Keys with binders} \label{sec:binders}

Thus far we have usefully consolidated the state of the art, but have not really done
anything new.  Tries are well known, and there are a number of papers about
tries in Haskell \cite{hinze etc}.  However, none of these works deal with keys that contain
binders, and that should be insensitive to alpha-conversion.  That is the challenge we
address next.  Here is our data type
\begin{code}
data Expr = App Expr Expr | Lam Var Expr | Var Var
\end{code}
The key idea is simple: we perform de-Bruijn numbering on the fly,
renamign each binder to a natural number, from outside in.
So, when inserting or looking up a key $(\lambda x.\, foo~ (\lambda y.\, x+y))$ we
behave as if the key was $(\lambda.\, foo ~(\lambda. \bv{1} + \bv{2}))$, where
each $\bv{i}$ stands for an occurrence of the variable bound by the $i$'th lambda.
In effect, then, we behave as if the data type was like this:
\begin{spec}
data Expr' = App Expr Expr | Lam Expr | FreeVar Var | BoundVar Int
\end{spec}
Notice (a) the |Lam| node no longer has a binder and (b) there are
two sorts of |Var| nodes, one for free variables and one for bound
variables. We will not actually build a value of type |Expr'| and look
that up in a trie keyed by |Expr'|; rather,
we are going to \emph{behave as if we did}. Here is the code
\begin{code}
data ExprMap v = EM { em_app :: ExprMap (ExprMap v)
                    , em_lam :: ExprMap v
                    , em_fv  :: Map Var v           -- Free variables
                    , em_bv  :: Map BoundVarKey v } -- Lambda-bound variables

lookupExpr :: Expr -> ExprMap v -> Maybe v
lookupExpr e = lkExpr (DB emptyBVM e)

data DBExpr = DB { edb_bvm :: BoundVarMap, edb_expr :: Expr }

lkExpr :: DBExpr -> ExprMap v -> Maybe v
lkExpr (DB bvm (App e1 e2)) = em_app >.> lkExpr (DB bvm e1) >=> lkExpr (DB bvm e2)
lkExpr (DB bvm (Lam v e))   = em_lam >.> lkExpr (DB (extendBVM v bvm) e)
lkExpr (DB bvm (Var v))     = case lookupBVM v bvm of
                                Nothing -> em_fv  >.> Map.lookup v  -- Free
                                Just bv -> em_bv  >.> Map.lookup bv -- Lambda-bound

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
that maps each lambda-bound variable to its de-Bruijn index, of type |BoundVarKey|.
\sg{We are using de Bruijn \emph{levels}, not \emph{indices}. A Var occurrence
with De Bruijn indices would count the number of lambdas between the occ and
its binding site. Levels, on the other hand, count the number of lambdas from
the root of the expr to the binding site. So within the const function
$\lambda x. \lambda y. y$, the occ of $y$ has DB index 0, but DB level 1.
Indices are favorable when doing subsitution which I presume we don't. See
also \url{https://randall-holmes.github.io/Watson/babydocs/node26.html} and
\url{https://mail.haskell.org/pipermail/haskell-cafe/2007-May/025424.html}.}
The key we look up --- the first argument of |lkExpr| --- becomes
a |DBExpr|, which is a pair of a |BoundVarMap| and an
|Expr|.
At a |Lam|
node we extend the |BoundVarMap|. At a |Var| node we
look up the variable in the |BoundVarMap| to decide whether it is
lambda-bound (within the key) or free, and behave appropriately.
The code for |alter| and |fold| holds no new surprises.
The construction of \Cref{sec:generalised}, to handle empty and singleton maps,
applies without difficulty to this generalised map.

And that is really all there is to it.  We regard this as a non-obvious merit
of the entire trie approach: it is quite remarkably easy to extend the basic
trie idea to be insensitive to alpha-conversion.

\section{Tries that match}

Next, we extend our tries to accomodate \emph{matching}, as we
sketched in \Cref{sec:matching-intro}.  A key advantage of tries over other representations is
that they can naturally extend to support matching.

\subsection{What ``matching'' means} \label{sec:matching-spec}

First, we have to ask what the API should be.
Our overall goal is to build a \emph{matching trie} into which we can:
\begin{itemize}
\item \emph{Insert} (pattern, value) pairs
\item \emph{Look up} a target expression, and return all the values whose pattern \emph{matches} that expression.
\end{itemize}
Semantically, then, a matching trie can be thought of as a set of (pattern, value) pairs.
What is a pattern? It is a pair $(vs,p)$ where
\begin{itemize}
\item $vs$ is a set of \emph{pattern variables}, such as $[a,b,c]$.
\item $p$ is a \emph{pattern expression}, such as $f\, a\, (g\, b\, c)$.
\end{itemize}
A pattern may of course contain free variables (not bound by the pattern), such as $f$ and $g$
in the above example. \sg{Which are regarded the same as constants by the algorithm. Correct?}
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
but not $(f~ 1~ (g~ v))$.  This ability important if we are to use matching tries
to implement class or type-family look in GHC.

It is sometimes desirable to be able to look up the \emph{most specific match} in the matching trie.  For example, suppose the matching trie contains
$$
\{ ([a],\, f\, a), ([p,q],\, f\,(p+q)) \}
$$
and suppose we look up $(f\,(2+x))$ in the trie.  The first entry matches, but the second also matches (with $S = [p \mapsto 2, q \mapsto x]$), and \emph{the second pattern is a substitution instance of the first}.  So we may want to return just the second match.  We call this \emph{most-specific matching}.

\subsection{The API of a matching trie} \label{sec:match-api}

Here are the signatures of the lookup and insertion\footnote{We begin with |insert|
  because it is simpler than |alter|} functions:
\rae{I changed |Expr| to |ExprS| here. Correct?}
\begin{code}
type ExprPat = ([PatVar], ExprS)
type PatVar  = Var
type Match v = ([(PatVar, ExprS)], v)

insertMExpr :: ExprPat -> v -> MExprMap v -> MExprMap v
lookupMExpr :: ExprS -> MExprMap v -> Bag (Match v)
\end{code}
\rae{What is the actual definition of |MExprMap|?}
A |MExprMap| is a trie, keyed by |Expr| \emph{patterns}.
A pattern variable, of type |PatVar| is just a |Var|; we
use the type synonym just for documentation purposes. When inserting into a
|MExprMap| we supply a pattern expression paired with the |[PatVar]|
over which the pattern is quantified.  When looking up in the map we return a \emph{bag}
of results (because more than one pattern might match).  Each item in this bag is
a |Match| that includes the |(PatVar, Expr)| pairs obtained by
matching the pattern, plus the value in the map (which presumably mentions those
pattern variables).

\sg{Why does lookupMExpr return a Bag? I thought we care for most-specific
matches? Shouldn't it then return a DAG of matches, or a tree, or at least a
list? Bag means no order at all... Later code assumes we can call map on Bags,
but Bag isn't defined anywhere. Maybe just return a list?
(A long time later, after I read 5.7) Ah, so it really is unordered. Fair
enough, but it would help to say as much.}

\subsection{Canonical patterns and pattern keys}

In \Cref{sec:binders} we saw how we could use de-Bruijn numbers to
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
that gives a list of (pattern-variable, value) pairs $[(x, e_1), (y,e_2)]$.
\sg{What is the difference between ,,expression'' and ,,value''
here? I think for a Match, you have to give the value in addition to the list of
pairs.}

Somehow we must accumulate a \emph{pattern-key map} that, for each
individual entry, maps its pattern keys back to its corresponding
pattern variables.  The pattern-key map is just a list of (pattern-variable, pattern-key) pairs.
For our example the pattern key map would be
$[(x, \pv{1}), (y,\pv{2})]$.  We can store the pattern key
map paired with the value, so that once we find a successful match we can use the pattern
key map and the pattern-key substitution to recover the pattern-variable substition that we want.

To summrise, suppose we want to build a matching trie for the following (pattern, value) pairs:
$$
(([x,y],\; f\;y\;(g\;y\;x)),\; v_1) \qquad and \qquad (([a],\; f\;a\;True),\;v_2)
$$
Then we will build a trie for the followng key-value pairs
$$
( (f \;\pv{}\;(g\;\pvo{1}\;\pv{})),\; (([(x,\pv{2}),(y,\pv{1})]), v_1) )
  \qquad and \qquad
( (f \;\pv{}\;True),\; ([(a,\pv{1})],\;v_2) )
$$


\subsection{Implementation: lookup} \label{sec:matching-lookup}

We are finally ready to give an implementation of matching tries.
We begin with |ExprS| (defined in \Cref{sec:ExprS}) as our key type;
that is we will not deal with lambdas and lambda-bound variables for now.
\Cref{sec:binders} will apply with no difficulty, but we can add that back
in after we have dealt with matching.

With these thoughts in mind, our matching trie has this definition:
\begin{code}
type PatKeys     = [(PatVar,PatKey)]
type MExprSMap v = MExprSMapX (PatKeys, v)

data MExprSMapX v
    = MM { mm_app  :: MExprSMap (MExprSMap v)
         , mm_fvar :: Map Var v
         , mm_pvar :: Maybe v     -- First occurrence of a pattern var
         , mm_xvar :: PatOccs v   -- Subsequent occurrence of a pattern var
             -- SG: I propose to combine |mm_pvar, mm_xvar| and have a single
             -- |mm_pvar :: PatKeyMap v|. We can resolve first (flex) and second
             -- or subseq (rigid) occs as we go.
       }
     | EmptyMM
type PatOccs v = [(PatKey,v)]
\end{code}
The client-visible |MExprSMap| with values of type |v|
is a matching trie |MExprSMapX| with values of type |(PatKeys,v)|,
as described in \Cref{sec:patkeymap}.
The trie |MExprSMapX| has four fields, one for each case in the pattern.
The first two fields deal with literals and applications, just as before. The third deals with the \emph{binding} occurrence
of a pattern variable $\pv{}$, and the fourth with a \emph{bound} occurrence of
a pattern variable $\pvo{i}$.

\sg{I think that most queries will have multiple Apps at the top-level and the
head will be a variable. It is a shame that we have to chase n pointers for an
n-ary application to get to the head! I think it would would be much more
efficient to store the list of App args in mm\_fvar, like}
\begin{spec}
         , mm_fvar :: Map Var (ListMap ExprS v)
\end{spec}
\sg{(NB: PatVars can't occur in app heads that way. If we want them to, we have to give
mm\_pvar a similar treatment.) IIRC, that is what Lean's DiscTree does, and it's
also how we implement RULE matching (grep for ru\_fn)...}

The core lookup function looks like this:
%{
%if style == newcode
%format lkMExprS = "lkMExprS0"
%endif
\begin{code}
lkMExprS :: forall v. ExprS -> (PatSubst, MExprSMapX v) -> Bag (PatSubst, v)

type PatKey = Int
data PatSubst = PS { ps_next  :: PatKey, ps_subst :: Map PatKey ExprS }

-- \rae{describe? omit?}
emptyPatSubst :: PatSubst
emptyPatSubst = PS { ps_next = 0, ps_subst = Map.empty }

extendPatSubst :: ExprS -> PatSubst -> PatSubst
extendPatSubst e (PS { ps_next = next, ps_subst = subst })
  = PS { ps_next = next + 1, ps_subst = Map.insert next e subst }
\end{code}
%}
As well as the target expression |ExprS| and the trie, the lookup function also takes
a |PatSubst| that gives the bindings for pattern variable bound so far.
It returns a bag of results, since more than one entry in the trie may match,
each paired with the |PatSubst| that binds the pattern variables.

Given |lkMExprS| we can write |lookupMExpr|,
the externally-callable lookup function:
\begin{code}
lookupMExprS :: ExprS -> MExprSMap v -> Bag (Match v)
lookupMExprS e m = fmap rejig (lkMExprS e (emptyPatSubst, m))
  where
    rejig :: (PatSubst, (PatKeys, v)) -> Match v
    rejig (ps, (pkmk, v)) = (map (lookupPatKey ps) pkmk, v)

lookupPatKey :: PatSubst -> (PatVar,PatKey) -> (PatVar,ExprS)
lookupPatKey subst (pat_var, pat_key) = (pat_var, lookupPatSubst pat_key subst)

lookupPatSubst :: PatKey -> PatSubst -> ExprS
lookupPatSubst pat_key (PS { ps_subst = subst })
  = case Map.lookup pat_key subst of
      Just expr -> expr
      Nothing   -> error "Unbound key"
\end{code}
Here |lookupMExpr| is just an impedence-matching shim around
a call to |lkMExprS| that does all the work.  Notice that the
input.  The latter returns a bag of |(PatSubst, (PatKeys, v))|
values, which the function |rejig| converts into the
the |Match v| results that we want.  The ``unbound key''
failure case in |lookupPatSubst| means that
|PatKeys| in a looked-up value asks for a key that is not
bound in the pattern.  The insertion function will ensure that this
never occurs.

Now we can return to the recursive function that does all the work: |lkMExprS|:
\begin{code}
lkMExprS :: forall v. ExprS -> (PatSubst, MExprSMapX v) -> Bag (PatSubst, v)
lkMExprS e (psubst, mt)
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
        VarS x     -> case Map.lookup x (mm_fvar mt) of
                        Just v  -> Bag.single (psubst,v)
                        Nothing -> Bag.empty
        AppS e1 e2 -> undefined -- @(psubst, mm_app mt) |> lkMExprS e1 >=> lkMExprS e2@  \rae{type error here}
\end{code}
The bag of results is the union of three possibilities, as follows. (Keep in mind that a |MExprSMap| represents \emph{many} patterns simultaneously.)
\begin{itemize}
\item |pat_var_bndr|: we consult the |mm_pvar|, if it contains |Just v| then at least one of the patterns in this trie has a pattern binder $\pv{}$ at this spot.  In that case we can simply bind the next free pattern variable (|ps_next|) to |e|, and return a singleton bag.
\item |pat_var_occs|: any of the bound pattern varaibles might have an occurrence $\pvo{i}$ at this spot, and a list of such bindings is held in |pat_var_occs|.  For each, we must do an equality check between the target |e| and the expression bound to that pattern variable (found via |lookupPatSubst|).  We return a bag of all values for which the equality check succeeds.
  \item |look_at_e| corresponds exactly to the cases we saw before in \Cref{sec:ExprS}.   The only subtlety is that we are are returning a \emph{bag} of results, but happily the Kleisli composition operator |(>=>)| (\Cref{fig:library}) works for any monad, including bags.
\end{itemize}

\subsection{Altering a matching trie}

\simon{Too much code, I know; but this section is one of the key contributions of the paper.}

How did the entries in our map get their |PatKeys|?  That
is, of course, the business of |insert|, or more generally
|alter|.  The key, recursive function must carry inwards a mapping
from pattern variables to pattern keys; we can simply re-use |BoundVarMap|
from \Cref{sec:bvm} for this purpose.  The exact signature for the function takes
a bit of puzzling out, and is worth comparing with its predecessor in \Cref{sec:alter}:
\begin{code}
type PatKeyMap = BoundVarMap   -- We re-use BoundVarMap

xtMExprS :: Set PatVar -> ExprS -> (PatKeyMap -> XT a)
         -> PatKeyMap -> MExprSMapX v -> MExprSMapX v
\end{code}
It is unsurprising the the function is given the set of pattern variables, so that it
can distinguish pattern variables from free variables.  It also takes a |PatKeyMap|, the
current binding of already-encountered pattern variables to their pattern keys;
when it completes the lookup it passes that completed binding map to the ``alter'' function.

Given this workhorse, we can build the client-visible |insert| function\footnote{|alter| is not much harder.}:
\begin{code}
insertExprS :: forall v. [Var]     -- Pattern variables
                         -> ExprS  -- Pattern
                         -> v -> MExprSMap v -> MExprSMap v
insertExprS pat_vs e v mm
  = xtExprS (Set.fromList pat_vs) e xt emptyBVM mm
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
It does so using the |PatKeyMap| accumulated by |xtExprS| and
finally passed to the local function |xt|.

Now we can define the workhorse, |xtExprS|:
\begin{code}
xtExprS pvs e xt pkm mm
  = case e of
      AppS e1 e2 -> mm { mm_app = xtMExprS pvs e1 (liftXTS (xtMExprS pvs e2 xt))
                                           pkm (mm_app mm) }

      VarS x | Just xv <- lookupBVM x pkm
             -> -- Second or subsequent occurrence of a pattern variable
                mm { mm_xvar = xtPatVarOcc xv (xt pkm) (mm_xvar mm) }

             | x `Set.member` pvs
             -> -- First occurrence of a pattern variable
                mm { mm_pvar = xt (extendBVM x pkm) (mm_pvar mm) }

             | otherwise
             -> -- A free variable
                mm { mm_fvar = Map.alter (xt pkm) x (mm_fvar mm) }

liftXTS :: (PatKeyMap -> MExprSMap v -> MExprSMap v)
        -> PatKeyMap -> Maybe (MExprSMap v) -> Maybe (MExprSMap v)
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
to return only the most-specific matches.  We simply modify the definition |lkMExprS| from \Cref{sec:matching-lookup} as follows:
%{
%if style == poly
%format as_before = "\ldots \text{as before} \ldots"
%else
%format as_before = "pat_var_occs :: Bag (PatSubst, v); pat_var_occs = undefined; look_at_e :: Bag (PatSubst, v); look_at_e = undefined; pat_var_bndr :: Bag (PatSubst, v); pat_var_bndr = undefined"
%format lkMExprS = "lkMExprS2"
%endif
\begin{code}
lkMExprS e (psubst, mt)
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

\section{Related work}

\begin{itemize}
\item Using a FSM; e.g \emph{Interpreted Pattern Match Execution} by Jeff Niu, a UW undergrad intern at Google.  https://docs.google.com/presentation/d/1e8MlXOBgO04kdoBoKTErvaPLY74vUaVoEMINm8NYDds/edit?usp=sharing

\item Matching multiple strings.
\end{itemize}

There is rich literature on radix trees, which incorporate the Singleton optimisation simply as ``each node that is the only child is merged with its parent'', and an abundance of related work in the theorem proving community available under the term ``Discrimination Tree'' and ``Term Indexing''. I think it would help the paper if instead of starting from ``an API for finite maps''/tries as a baseline, it would start from "an API for term indexing"/discrimination trees as decribed in the Handbook of Automated Reasoning (2001), for example. I'll have access to a hard copy in a couple of days and can then report on the contents...

Here's a GH issue that suggests using Discrimination Trees to speed up Hoogle queries: https://github.com/ndmitchell/hoogle/issues/250. That thread generally seems like a good source of references to consider. It suggests that discrimination trees are but the simplest data structure to perform term indexing.
Remy Goldschmidt (@@taktoa, the GH issue creator) even provides a model implementation of discrimination trees in Haskell: https://gist.github.com/taktoa/7a4d77ebc3a312dd69bb19199d30863b

Here's a paper from 1994 claiming to be faster than discrimination trees: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.43.823 (In case you're wondering, I'm not affiliated with the author at all.)

It appears that at least since 2009, the 1994 approach has already been extended to a higher-order pattern scenario (allowing e.g. miller pattern unification): https://dl.acm.org/doi/10.1145/1614431.1614437

\end{document}
