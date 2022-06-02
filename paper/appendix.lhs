%include rae.fmt
\let\restriction\relax

\section{Evaluation} \label{sec:eval-extended}

So far, we have seen that trie maps offer a significant advantage over other
kinds of maps like ordered maps or hash maps: the ability to do a matching
lookup (in \Cref{sec:matching}). In this section, we will see that query
performance is another advantage. Our implementation of trie maps in Haskell
can generally compete in performance with other map data structures, while
significantly outperforming traditional map implementations on some operations.
Not bad for a data structure that we can also extend to support matching lookup!

\subsection{Runtime} \label{sec:runtime}

\begin{table}

  \caption{Benchmarks of different operations over our trie map |ExprMap| (TM),
  ordered maps |Map Expr| (OM) and hash maps |HashMap Expr| (HM), varying the
  size parameter $N$.  Each map is of size $N$ (so $M=N$) and the expressions
  it contains are also each of size $N$ (so $E=N$).
  We give the measurements of OM and HM relative to absolute runtime
  measurements for TM. Lower is better. Digits whose order of magnitude is
  no larger than that of twice the standard deviation are marked by squiggly
  lines.}
  \begin{tabular}{l rrr rrr rrr}
  \toprule
  $N$  & \multicolumn{3}{c}{\textbf{10}} & \multicolumn{3}{c}{\textbf{100}} & \multicolumn{3}{c}{\textbf{1000}} \\
  \cmidrule(lr{.5em}){2-4} \cmidrule(lr{.5em}){5-7} \cmidrule(lr{.5em}){8-10}
  Data structure & TM & OM & HM
                 & TM & OM & HM
                 & TM & OM & HM \\
  \midrule
  % input without {}, so that we get the primitive TeX input.
  % See https://tex.stackexchange.com/questions/567985/problems-with-inputtable-tex-hline-after-2020-fall-latex-release
  \input bench-overview.tex-incl
  \bottomrule
  \end{tabular}

  \label{fig:runtime}
\end{table}

We measured the runtime performance of the (non-matching) |ExprMap| data
structure on a selection of workloads, conducted using the \hackage{criterion}
benchmarking library%
\footnote{The benchmark machine runs Ubuntu 18.04 on an Intel Core i5-8500 with
16GB RAM. All programs were compiled with \texttt{-O2 -fproc-alignment=64} to
eliminate code layout flukes and run with \texttt{+RTS -A128M -RTS} for 128MB
space in generation 0 in order to prevent major GCs from skewing the results.}.
\Cref{fig:runtime} presents an overview of the results, while
\Cref{fig:runtime-finer} goes into more detail on some configurations.

\subsubsection*{Setup}
All benchmarks except the \benchname{fromList*} variants are handed a pre-built
map containing $N$ expressions, each consisting of roughly $N$ |Expr| data
constructors, and drawn from a pseudo-random source with a fixed (and thus
deterministic) seed. $N$ is varied between 10 and 1000.

We compare three different non-matching map implementations, simply because we
were not aware of other map data structures with matching lookup modulo
$\alpha$-equivalence and we wanted to compare apples to apples.
The |ExprMap| forms the baseline. Asymptotics are given with respect to map
size $n$ and key expression size $k$:

\begin{itemize}
  \item |ExprMap| (designated ``TM'' in \Cref{fig:runtime}) is the trie map
        implementation from this paper. Insertion and lookup and have to perform
        a full traversal of the key, so performance should scale with
        $\mathcal{O}(k)$, where $k$ is the key |Expr| that is accessed.
  \item |Map Expr| (designated ``OM'') is the ordered map implementation from
        the mature, well-optimised \hackage{containers} library. It uses size
        balanced trees under the hood \cite{adams}. Thus, lookup and insert
        operations incur an additional log factor in the map size $n$, for a
        total of $\mathcal{O}(k \log n)$ factor compared to both other maps.
  \item |HashMap Expr| (designated ``HM'') is an implementation of hash array
        mapped tries \cite{hamt} from the \hackage{unordered-containers}
        library. Like |ExprMap|, map access incurs a full traversal of the key
        to compute a hash and then a $\mathcal{O}(\log_{32} n)$ lookup in the
        array mapped trie. The log factor can be treated like a constant for all
        intents and purposes, so lookup and insert is effectively in
        $\mathcal{O}(k)$.
\end{itemize}
Benchmarks ending in \benchname{\_lam}, \benchname{\_app1}, \benchname{\_app2}
add a shared prefix to each of the expressions before building the initial
map:
\begin{itemize}
  \item \benchname{\_lam} wraps $N$ layers of |(Lam "$")| around each expression
  \item \benchname{\_app1} wraps $N$ layers of |(Var "$"^^^`App`)| around each expression
  \item \benchname{\_app2} wraps $N$ layers of |(`App`^^^Var "$")| around each expression
\end{itemize}
where |"$"| is a name that doesn't otherwise occur in the generated expressions.

\begin{itemize}
  \item The \benchname{lookup\_all*} family of benchmarks looks up every
        expression that is part of the map. So for a map of size 100, we
        perform 100 lookups of expressions each of which have approximately size
        100. \benchname{lookup\_one} looks up just one expression that is
        part of the map.
  \item \benchname{insert\_lookup\_one} inserts a random expression into the
        initial map and immediately looks it up afterwards. The lookup is to
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
\benchname{\_app1} variants show that |ExprMap| can win substantially against
an ordered map representation: |ExprMap| looks at the shared prefix exactly
once one lookup, while |Map| has to traverse the shared prefix of length
$\mathcal{O}(N)$ on each of its $\mathcal{O}(\log N)$ comparisons. As
a result, the gap between |ExprMap| and |Map| widens as $N$ increases,
confirming an asymptotic difference. The advantage is less pronounced in
the \benchname{\_app2} variant, presumably because |ExprMap| can't share
the common prefix here: it turns into an unsharable suffix in the pre-order
serialisation, blowing up the trie map representation compared to its sibling
\benchname{\_app1}.

Although |HashMap| loses on most benchmarks compared to |ExprMap| and |Map|, most
measurements were consistently at most a factor of two slower than |ExprMap|.
We believe that is due to the fact that it is enough to traverse the |Expr| once
to compute the hash, thus it is expected to scale similarly as |ExprMap|.

Comparing the \benchname{lookup\_all*} measurements of the same map data
structure on different size parameters $N$ reveals a roughly quadratic correlation
throughout all implementations, give or take a logarithmic factor.
That seems plausible given that $N$ linearly affects expression size and map
size (and thus, number of lookups). But realistic workloads tend to have much
larger map sizes than expression sizes!

\begin{table}
  \centering
  \caption{Varying expression size $E$ and map size $M$ independently on benchmarks
  \benchname{lookup\_all} and \benchname{insert\_lookup\_one}.}
  \resizebox{\textwidth}{!}{%
    \begin{tabular}{cr rrr rrr rrr rrr}
    \toprule
    \multicolumn{2}{c}{\multirow{2}{*}{\diagbox{$E$}{$M$}}} & \multicolumn{3}{c}{\textbf{10}}
                                        & \multicolumn{3}{c}{\textbf{100}}
                                        & \multicolumn{3}{c}{\textbf{1000}}
                                        & \multicolumn{3}{c}{\textbf{10000}} \\
    \cmidrule(lr{.5em}){3-5} \cmidrule(lr{.5em}){6-8} \cmidrule(lr{.5em}){9-12} \cmidrule(lr{.5em}){12-14}
                       & & TM & OM & HM
                         & TM & OM & HM
                         & TM & OM & HM
                         & TM & OM & HM \\
    \midrule
    \multirow{4}{*}{\rotatebox{90}{\benchname{lookup\_all}}}
    \input bench-lookup_all.tex-incl
    \midrule
    \multirow{4}{*}{\rotatebox{90}{\benchname{lo\_a\_app1}}}
    \input bench-lookup_all_app1.tex-incl
    \midrule
    \multirow{4}{*}{\rotatebox{90}{\benchname{insert\_o\_l}}}
    \input bench-insert_lookup_one.tex-incl
    \midrule
    \multirow{4}{*}{\rotatebox{90}{\benchname{fromList }}}
    \input bench-fromList.tex-incl
    \midrule
    \multirow{4}{*}{\rotatebox{90}{\benchname{union}}}
    \input bench-union.tex-incl
    \bottomrule
    \end{tabular}
  }

  \label{fig:runtime-finer}
\end{table}

Let us see what happens if we vary map size $M$ and expression
size $E$ independently for \benchname{lookup\_all}. The results in
\Cref{fig:runtime-finer} show that |ExprMap| scales better than |Map| when we
increase $M$ and leave $E$ constant. The difference is even more pronounced than
in \Cref{fig:runtime}, in which $N = M = E$.

The time measurements for |ExprMap| appear to grow almost linearly with $M$.
Considering that the number of lookups also increases $M$-fold, it seems the
cost of a single lookup remained almost constant, despite the fact that we store
varying numbers of expressions in the trie map. That is exactly the strength
of a trie implementation: Time for the lookup is in $\mathcal{O}(E)$, i.e.,
linear in $E$ but constant in $M$. The same does not hold for search trees,
where lookup time is in $\mathcal{O}(P \log M)$. $P \in \mathcal{O}(E)$ here and
captures the common short circuiting semantics of the lexicographic order on
|Expr|. It denotes the size of the longest shared prefix of all expressions.

By contrast, fixing $M$ but increasing $E$ makes |Map| easily catch up
on lookup performance with |ExprMap|, ultimately outpacing it. The shared prefix
factor $P$ for |Map| remains essentially constant relative to $E$: larger
expressions still are likely to differ very early because they are random.
Increasing $M$ will introduce more clashes and is actually more likely to
increase $P$ on completely random expressions. As written above, realistic
work loads often have shared prefixes like \benchname{lookup\_all\_app1}, where
we already saw that |ExprMap| outperforms |Map|. The fact that |Map| performance
depends on $P$ makes it an extremely workload dependent pick, leading to
compiler performance that is difficult to predict. |HashMap| shows performance
consistent with |ExprMap| but is a bit slower, as before. There is no subtle
scaling factor like $P$; just plain predictable $\mathcal{O}(E)$ like |ExprMap|.

Returning to \Cref{fig:runtime}, we see that folding over |ExprMap|s is
considerably slower than over |Map| or |HashMap|. The complex tree structure is
difficult to traverse and involves quite a few indirections.
This is in stark contrast to the situation with |Map|, where it's just a
textbook in-order traversal over the search tree. Folding over |HashMap|
performs similarly to |Map|.

We think that |ExprMap| folding performance dies by a thousand paper cuts: The
lazy fold implementation means that we allocate a lot of thunks for intermediate
results that we end up forcing anyway in the case of our folding operator |(+)|.
That is a price that |Map| and |HashMap| pay, too, but not nearly as much as the
implementation of |foldrEM| does.
Furthermore, there's the polymorphic recursion in the head case of |em_app|
with a different folding function |(foldrTM f)|, which allocates on each call
and makes it impossible to specialise |foldrEM| for a fixed folding function
like |(+)| with the static argument transformation~\cite{santos}. Hence
we tried to single out the issue by ensuring that |Map| and |ExprMap| in
fact don't specialise for |(+)| when running the benchmarks, by means of a
\texttt{NOINLINE} pragma.
Another possible reason might be that the code generated for |foldrEM| is quite
a lot larger than the code for |Map|, say, so we are likely measuring caching
effects.
We are positive there are numerous ways in which the performance of |foldrEM|
can be improved, but believe it is unlikely to exceed or just reach the
performance of |Map|.

\subsubsection*{Building}
The \benchname{insert\_lookup\_one} benchmark demonstrates that |ExprMap| also
wins on insert performance, although the defeat against |Map| for size
parameters beyond 1000 is looming. Again, \Cref{fig:runtime-finer} decouples
map size $M$ and expression size $E$. The data suggests that in comparison to
|Map|, $E$ indeed affects insert performance of |ExprMap| linearly. By contrast,
$M$ does not seem to affect insert performance at all.

The small edge that |ExprMap| seems to have over |Map| and |HashMap|
doesn't carry over to its naïve |fromList| implementation, though. |Map| wins
the \benchname{fromList} benchmark, albeit with |ExprMap| as a close second.
That is a bit surprising, given that |Map|'s |fromList| quickly falls back to a
list fold like |ExprMap| on unsorted inputs, while |HashMap| has a less naïve
implementation: it makes use of transient mutability and performs destructive
inserts on the map data structure during the call to |fromList|, knowing that
such mutability can't be observed by the caller. Yet, it still performs worse
than |ExprMap| or |Map| for larger $E$, as can be seen in
\Cref{fig:runtime-finer}.

% % Not an issue anymore with -A128M
% Cursory investigation of |ExprMap| suggests that it spends much more time in
% garbage collection than |Map|; over the course of a test program reproducing
% the $N = 1000$ case, the generational garbage collector had to copy more than
% thrice as many bytes. One reason is that the (fixed) input list of
% expressions will quickly end up in the old generation and |Map|'s internal nodes
% simply store pointers to those expressions. That produces much less short-lived
% garbage than |ExprMap|, which allocates one large |EM| constructor for each
% |Expr| node in the shared prefix. If a minor GC is triggered during the call to
% |fromList|, some of the garbage will end up in the old generation, resulting in
% more costly major GCs. |criterion| performs a minor GC before each measurement
% of a benchmark, so we tried to increase the size of the young generation in
% order to prevent intermittent major GC during measurements. It appears that
% brings perf on par with |Map|, but still doesn't win.

We expected |ExprMap| to take the lead in \benchname{fromList\_app1}. And indeed
it does, outperforming |Map| for larger $N$ which pays for having to compare the
shared prefix repeatedly. But |HashMap| is good for another surprise and keeps
on outperforming |ExprMap| for small $N$.

What would a non-naïve version of |fromList| for |ExprMap| look like? Perhaps
the process could be sped up considerably by partitioning the input list
according to the different fields of |ExprMap| like |em_lam| and then calling
the |fromList| implementations of the individual fields in turn. The process
would be very similar to discrimination sort~\cite{discr-sort}, which is a
generalisation of radix sort to tree-like data and very close to tries.
Indeed, the \hackage{discrimination} library provides such an optimised
$\mathcal{O}(N)$ |toMap| implementation for ordered maps.

The \benchname{union*} benchmarks don't reveal anything new; |Map| and |HashMap|
win for small $N$, but |ExprMap| wins in the long run, especially when there's
a sharable prefix involved.

\subsection{Space}

\begin{table}
  \centering
  \caption{Varying expression size $E$ and map size $M$ while measuring the
  memory footprint of the different map implementions on 4 different expression
  populations. Measurements of |Map| (OM) and |HashMap| (HM) are displayed as
  relative multiples of the absolute measurements on |ExprMap| (TM). Lower is
  better. \dag indicates heap overflow.}
  \resizebox{\textwidth}{!}{%
    \begin{tabular}{rr rrr rrr rrr rrr}
    \toprule
    \multicolumn{2}{c}{\multirow{2}{*}{\diagbox{$E$}{$M$}}} & \multicolumn{3}{c}{\textbf{10}}
                                        & \multicolumn{3}{c}{\textbf{100}}
                                        & \multicolumn{3}{c}{\textbf{1000}}
                                        & \multicolumn{3}{c}{\textbf{10000}} \\
    \cmidrule(lr{.5em}){3-5} \cmidrule(lr{.5em}){6-8} \cmidrule(lr{.5em}){9-12} \cmidrule(lr{.5em}){12-14}
                       & & TM & OM & HM
                         & TM & OM & HM
                         & TM & OM & HM
                         & TM & OM & HM \\
    \midrule
    \multirow{4}{*}{\rotatebox{90}{\benchname{space}}}
    \input bench-space.tex-incl
    \midrule
    \multirow{4}{*}{\rotatebox{90}{\benchname{space\_app1}}}
    \input bench-space_app1.tex-incl
    \midrule
    \multirow{4}{*}{\rotatebox{90}{\benchname{space\_app2}}}
    \input bench-space_app2.tex-incl
    \midrule
    \multirow{4}{*}{\rotatebox{90}{\benchname{space\_lam}}}
    \input bench-space_lam.tex-incl
    \bottomrule
    \end{tabular}
  }

  \label{fig:space}
\end{table}

We also measured the memory footprint of |ExprMap| compared to |Map| and
|HashMap|. The results are shown in \Cref{fig:space}. All four benchmarks simply
measure the size on the heap in bytes of a map consisting of $M$ expressions of
size $E$. They only differ in whether or not the expressions have a shared
prefix. As before, \benchname{space} is built over completely random expressions,
while the other three benchmarks build maps with common prefixes, as discussed in
\cref{sec:runtime}.

In \benchname{space}, prefix sharing is highly unlikely for reasons discussed
in the last section: Randomness dictates that most expressions diverge quite
early in their prefix. As a result, |ExprMap| consumes slightly more space
than both |Map| and |ExprMap|, the latter of which wins every single instance.
The difference here is ultimately due to the fact that inner nodes in the trie
allocate more space than inner nodes in |Map| or |ExprMap|.

However, in \benchname{space\_app1} and \benchname{space\_lam}, we can see that
|ExprMap| is able to exploit the shared prefixes to great effect: For big
$M$, the memory footprint of \benchname{space\_app1} approaches that of
\benchname{space} because the shared prefix is only stored once. In the other
dimension along $E$, memory footprint still increases by similar factors as
in \benchname{space}. The \benchname{space\_lam} family does need a bit more
bookkeeping for the de Bruijn numbering, so the results aren't quite as close to
\benchname{space\_app1}, but it's still an easy win over |Map| and |HashMap|.

For \benchname{space\_app2}, |ExprMap| can't share any prefixes because the
shared structure turns into a suffix in the pre-order serialisation. As a result,
|Map| and |HashMap| allocate less space, all consistent constant factors apart
from each other. |HashMap| wins here again.

\section{Triemaps that unify?}

In effect, the |PatVar|s of the patterns stored in our matching triemaps act
like unification variables. The unification problems we solve are always
particularly simple, because pattern variables only ever match against are
\emph{expression} keys in which no pattern variable can occur.

Another frustrating point is that we had to duplicate the |TrieMap| class in
\Cref{sec:matching-class} because the key types for lookup and insertion no
longer match up. If we managed to generalise the lookup key from expressions to
patterns, too, we could continue to extend good old |TrieMap|.
All this begs the question: \emph{Can we extend our idiomatic triemaps to facilitate
unifying lookup?}

At first blush, the generalisation seems simple. We already carefully confined
the matching logic to |Matchable|. It should be possible to generalise to
\begin{code}
class (Eq k, MonadPlus (Unify k)) => Unifiable k where
  type Unify k :: Type -> Type
  unify :: k -> k -> Unify k ()
class (Unifiable (TrieKey tm), TrieMap tm) => UTrieMap tm where
  lookupUniUTM :: TrieKey tm -> tm v -> Unify (TrieKey tm) v
\end{code}
But there are problems:
\begin{itemize}
  \item We would need all unification variables to be globally unique lest we
    open ourselves to numerous shadowing issues when reporting unifiers.
  \item Consider the Unimap for
    $$
      (([a], T\;a\;A), v1) \quad \text{and} \quad (([b], T\;b\;B), v2)
    $$
    After canonicalisation, we get
    $$
      ((T\;\pv{1}\;A), ([(a,\pv{1})], v1)) \quad \text{and} \quad (T\;\pv{1}\;B, ([(b,\pv{1})], v2))
    $$
    and both patterns share a prefix in the trie.
    Suppose now we uni-lookup the pattern $([c,d], T c d)$.
    What should we store in our |UniState| when unifying $c$ with $\pv{1}$?
    There simply is no unique pattern variable to \enquote{decanonicalise} to!
    In general, it appears we'd get terms in the range of our substitution that
    mix |PatVar|s and |PatKey|s. Clearly, the vanilla |Expr| datatype doesn't
    accomodate such an extension and we'd have to complicate its definition with
    techniques such as Trees that Grow \cite{ttg}.
\end{itemize}

So while embodying full-blown unification into the lookup algorithm seems
attractive at first, in the end it appears equally complicated to present.


