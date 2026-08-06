# Sources and licensing for the demo corpus

Every text is out of copyright in the UK and the US. All authors died more than
100 years ago, and every edition used is one Project Gutenberg distributes as
public domain in the US.

| # | Work | Author | Died | Gutenberg |
|---|---|---|---|---|
| 1342 | Pride and Prejudice | Jane Austen | 1817 | [ebook 1342](https://www.gutenberg.org/ebooks/1342) |
| 1661 | The Adventures of Sherlock Holmes | Arthur Conan Doyle | 1930 | [ebook 1661](https://www.gutenberg.org/ebooks/1661) |
| 164 | Twenty Thousand Leagues under the Sea | Jules Verne | 1905 | [ebook 164](https://www.gutenberg.org/ebooks/164) |
| 2701 | Moby Dick | Herman Melville | 1891 | [ebook 2701](https://www.gutenberg.org/ebooks/2701) |
| 1228 | On the Origin of Species | Charles Darwin | 1882 | [ebook 1228](https://www.gutenberg.org/ebooks/1228) |
| 944 | The Voyage of the Beagle | Charles Darwin | 1882 | [ebook 944](https://www.gutenberg.org/ebooks/944) |
| 10136 | The Book of Household Management | Mrs Beeton | 1865 | [ebook 10136](https://www.gutenberg.org/ebooks/10136) |
| 2680 | Meditations | Marcus Aurelius | 180 | [ebook 2680](https://www.gutenberg.org/ebooks/2680) |

## The ids are asserted, not trusted

`build_gutenberg.py` checks each download's `Title:` **and** `Author:` lines
against the table above and aborts on a mismatch.

That check was added after the first version of this file was wrong twice over:
10681 is Roget's Thesaurus rather than Mrs Beeton, and the real *Voyage of the
Beagle* is 944 rather than 10136. Nothing about the output looked broken — the
pipeline clustered the thesaurus perfectly well, producing confident groups
around `lat, phr, est` and `adj, honor, lat` filed under "reference-cookery".

A thesaurus is also simply the wrong shape of text for this: it is word lists
rather than prose, so there is no topic for an embedding to find. It has been
dropped rather than recategorised.

The author check matters as much as the title, because titles are not unique —
"Meditations" is Marcus Aurelius here and Descartes elsewhere. Only the surname
is compared, since Gutenberg's Author line carries honorifics no sane table
entry would reproduce ("Emperor of Rome Marcus Aurelius").

## The Project Gutenberg trademark

Project Gutenberg's licence applies to its *trademark and its boilerplate*, not
to the underlying public-domain texts. `build_gutenberg.py` strips the header
and footer between the `*** START OF …***` and `*** END OF …***` markers, so
the corpus contains none of it and makes no use of the trademark.

This is not only a licensing matter. Left in place, the boilerplate is long,
formulaic and near-identical across all eight books — so it forms its own large
and perfectly coherent cluster about the Project Gutenberg Literary Archive
Foundation, which is a good demonstration of the pipeline working and a
completely useless one.

## Redistribution

The dumps published on this repository's releases contain only stripped
public-domain text plus vectors computed from it. They are offered under
[CC0](../LICENSE) along with the rest of the repository.
