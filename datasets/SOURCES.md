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
| 10136 | The Voyage of the Beagle | Charles Darwin | 1882 | [ebook 10136](https://www.gutenberg.org/ebooks/10136) |
| 10681 | The Book of Household Management | Isabella Beeton | 1865 | [ebook 10681](https://www.gutenberg.org/ebooks/10681) |
| 2680 | Meditations | Marcus Aurelius | 180 | [ebook 2680](https://www.gutenberg.org/ebooks/2680) |

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
