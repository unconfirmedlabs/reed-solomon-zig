use std::fmt;

// ======================================================================
// ERROR - PUBLIC

/// Error returned by [`rustdocify`].
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Version was given to [`rustdocify`] but URL is missing a version.
    ///
    /// # Example
    ///
    /// ```markdown
    /// [foo]: https://docs.rs/foo
    /// ```
    MissingVersionInUrl(String),

    /// Readme has top-level header that is not first header.
    ///
    /// # Example
    ///
    /// <!-- Note: Using extra `#`:s here because rustdoc removes one. -->
    /// ```markdown
    /// ### First
    /// ## Second
    /// ```
    NonFirstTopLevelHeader(String),

    /// URL is not recognized as valid.
    ///
    /// This means that either URL is invalid or this crate has a bug.
    ///
    /// # Example
    ///
    /// ```markdown
    /// [foo]: https://docs.rs/foo/*/foo/hello_world.html
    /// ```
    UnrecognizedUrl(String),

    /// Crate name was given to [`rustdocify`] but URL has different crate name.
    ///
    /// # Example
    ///
    /// ```markdown
    /// [foo]: https://docs.rs/foo/*/DIFFERENT_CRATE
    /// ```
    WrongCrateNameInUrl(String),

    /// Version was given to [`rustdocify`] but URL has different version.
    ///
    /// # Example
    ///
    /// ```markdown
    /// [foo]: https://docs.rs/foo/DIFFERENT_VERSION
    /// ```
    WrongVersionInUrl(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::MissingVersionInUrl(url) => {
                write!(f, "missing version in url: {}", url)
            }

            Error::NonFirstTopLevelHeader(header) => {
                write!(f, "non-first top level header: {}", header)
            }

            Error::UnrecognizedUrl(url) => {
                write!(f, "unrecognized url: {}", url)
            }

            Error::WrongCrateNameInUrl(url) => {
                write!(f, "wrong crate name in url: {}", url)
            }

            Error::WrongVersionInUrl(url) => {
                write!(f, "wrong version in url: {}", url)
            }
        }
    }
}

impl std::error::Error for Error {}

// ======================================================================
// FUNCTIONS - PUBLIC

/// Rustdocifies the given readme.
///
/// - Removes top-level header.
/// - Changes other headers to be one level higher.
/// - Converts `docs.rs` links of given `package_name` to rustdoc format.
/// - Doesn't change anything within code blocks.
/// - If `version` is given, checks that links have this exact version.
/// - If `crate_name` is given, checks that links have this exact crate name, if any.
///
/// See [crate index] for an example and more details.
///
/// [crate index]: crate
pub fn rustdocify(
    readme: &str,
    package_name: &str,
    version: Option<&str>,
    crate_name: Option<&str>,
) -> Result<String, Error> {
    let mut is_first_header = true;
    let mut code_block_level = None;

    let mut result = String::with_capacity(readme.len());

    for line in readme.split_inclusive('\n') {
        if let Some(level) = code_block_level {
            // IN CODE BLOCK

            if is_code_block_end(line, level) {
                code_block_level = None;
            }

            result.push_str(line);
        } else {
            // NOT IN CODE BLOCK

            code_block_level = is_code_block_start(line);

            if code_block_level.is_some() {
                result.push_str(line);
            } else if let Some(line) = convert_header_line(line, &mut is_first_header)? {
                result.push_str(line);
            } else if let Some(line) = convert_link_line(line, package_name, version, crate_name)? {
                result.push_str(&line);
            } else {
                result.push_str(line);
            }
        }
    }

    Ok(result)
}

// ======================================================================
// FUNCTIONS - PRIVATE

// Returns
// - `Ok(Some(..))` on successful conversion
// - `Ok(None)` if this is not a header line
// - `Err(..)` on error
fn convert_header_line<'a>(
    line: &'a str,
    is_first_header: &mut bool,
) -> Result<Option<&'a str>, Error> {
    let bytes = line.as_bytes();

    let mut level = 0;
    while level < line.len() && bytes[level] == b'#' {
        level += 1;
    }

    if level > 0 && bytes[level] == b' ' {
        if level == 1 {
            if *is_first_header {
                *is_first_header = false;
                Ok(Some(""))
            } else {
                Err(Error::NonFirstTopLevelHeader(line.to_owned()))
            }
        } else {
            *is_first_header = false;
            Ok(Some(&line[1..]))
        }
    } else {
        Ok(None)
    }
}

// [...]: https://docs.rs/PACKAGE...
//
// Returns
// - `Ok(Some(..))` on successful conversion
// - `Ok(None)` if this is not a link line
// - `Err(..)` on error
fn convert_link_line(
    line: &str,
    package_name: &str,
    version: Option<&str>,
    crate_name: Option<&str>,
) -> Result<Option<String>, Error> {
    let bytes = line.as_bytes();

    if bytes[0] != b'[' {
        return Ok(None);
    }

    let close_bracket_pos = if let Some(pos) = line.find(']') {
        pos
    } else {
        return Ok(None);
    };

    if bytes[close_bracket_pos + 1] != b':' {
        return Ok(None);
    }

    let url_start_pos =
        if let Some(pos) = line[close_bracket_pos + 2..].find(|c: char| !c.is_whitespace()) {
            close_bracket_pos + 2 + pos
        } else {
            return Ok(None);
        };

    if url_start_pos == close_bracket_pos + 2 {
        return Ok(None);
    }

    let url_end_pos = if let Some(pos) = line[url_start_pos..].find(|c: char| c.is_whitespace()) {
        url_start_pos + pos
    } else {
        line.len()
    };

    let url = &line[url_start_pos..url_end_pos];

    match convert_url(url, package_name, version, crate_name) {
        Ok(link) => Ok(Some(format!(
            "{}{}{}",
            &line[..url_start_pos],
            link,
            &line[url_end_pos..]
        ))),
        Err(error) => Err(error),
    }
}

fn convert_url(
    url: &str,
    package_name: &str,
    version: Option<&str>,
    crate_name: Option<&str>,
) -> Result<String, Error> {
    let url_prefix = format!("https://docs.rs/{}", package_name);
    if !url.starts_with(&url_prefix) {
        return Ok(url.to_owned());
    }

    // url_prefix + optional '/'
    let url_prefix_len = if url.len() == url_prefix.len() {
        url_prefix.len()
    } else {
        let byte_after_prefix = url.as_bytes()[url_prefix.len()];
        if byte_after_prefix == b'/' {
            url_prefix.len() + 1
        } else if byte_after_prefix == b'#' {
            url_prefix.len()
        } else {
            return Ok(url.to_owned());
        }
    };

    let mut path: Vec<&str>;
    let fragment;

    if let Some(fragment_start_pos) = url.find('#') {
        path = url[url_prefix_len..fragment_start_pos].split('/').collect();
        fragment = Some(&url[fragment_start_pos + 1..]);
    } else {
        path = url[url_prefix_len..].split('/').collect();
        fragment = None;
    }

    if path.last() == Some(&"") {
        path.pop();
    }

    // VERSION

    if path.is_empty() {
        // NO VERSION IN URL

        if version.is_some() {
            return Err(Error::MissingVersionInUrl(url.to_owned()));
        } else {
            return Ok(root_link(fragment));
        }
    }

    let url_version = path[0];

    if let Some(version) = version {
        if url_version != version {
            return Err(Error::WrongVersionInUrl(url.to_owned()));
        }
    }

    if path.len() == 2 && path[1] == "index.html" {
        return Ok(root_link(fragment));
    }

    // CRATE NAME

    if path.len() == 1 {
        // NO CRATE NAME IN URL

        return Ok(root_link(fragment));
    }

    let url_crate = path[1];

    if let Some(crate_name) = crate_name {
        if url_crate != crate_name {
            return Err(Error::WrongCrateNameInUrl(url.to_owned()));
        }
    }

    if path.len() == 3 && path[2] == "index.html" {
        return Ok(root_link(fragment));
    }

    // FILENAME

    if path.len() == 2 {
        // NO FILENAME IN URL

        return Ok(root_link(fragment));
    }

    let last = *path.last().unwrap();

    let (modules, filename) = if last.contains('.') {
        (&path[2..path.len() - 1], last)
    } else {
        (&path[2..], "index.html")
    };

    let modules = if modules.is_empty() {
        "".to_owned()
    } else {
        format!("::{}", modules.join("::"))
    };

    if filename == "index.html" {
        if let Some(fragment) = fragment {
            return Ok(format!("crate{}#{}", modules, fragment));
        } else {
            return Ok(format!("crate{}", modules));
        }
    }

    // enum.ENUM.html
    // enum.ENUM.html#method.METHOD
    // enum.ENUM.html#variant.VARIANT
    // enum.ENUM.html#fragment
    // fn.FUNCTION.html
    // fn.FUNCTION.html#fragment
    // macro.FUNCTION.html
    // macro.FUNCTION.html#fragment
    // struct.STRUCT.html
    // struct.STRUCT.html#method.METHOD
    // struct.STRUCT.html#fragment
    // trait.TRAIT.html
    // trait.TRAIT.html#tymethod.METHOD
    // trait.TRAIT.html#fragment

    const DATA: &[(&str, &[&str])] = &[
        ("enum.", &["method.", "variant."]),
        ("fn.", &[]),
        ("macro.", &[]),
        ("struct.", &["method."]),
        ("trait.", &["tymethod."]),
    ];

    if !filename.ends_with(".html") {
        Err(Error::UnrecognizedUrl(url.to_owned()))
    } else {
        for (prefix, special_fragment_starts) in DATA {
            if filename.starts_with(prefix) {
                let name = &filename[prefix.len()..&filename.len() - 5];
                if let Some(fragment) = fragment {
                    for special_fragment_start in special_fragment_starts.iter() {
                        if let Some(fragment_name) = fragment.strip_prefix(special_fragment_start) {
                            return Ok(format!("crate{}::{}::{}", modules, name, fragment_name));
                        }
                    }
                    return Ok(format!("crate{}::{}#{}", modules, name, fragment));
                } else {
                    return Ok(format!("crate{}::{}", modules, name));
                }
            }
        }

        Err(Error::UnrecognizedUrl(url.to_string()))
    }
}

fn is_code_block_end(line: &str, backtick_count: usize) -> bool {
    if line.len() < backtick_count {
        false
    } else {
        for n in 0..backtick_count {
            if line.as_bytes()[n] != b'`' {
                return false;
            }
        }
        true
    }
}

fn is_code_block_start(line: &str) -> Option<usize> {
    let mut backtick_count = 0;
    while backtick_count < line.len() && line.as_bytes()[backtick_count] == b'`' {
        backtick_count += 1;
    }

    if backtick_count >= 3 {
        Some(backtick_count)
    } else {
        None
    }
}

fn root_link(fragment: Option<&str>) -> String {
    if let Some(fragment) = fragment {
        format!("crate#{}", fragment)
    } else {
        "crate".to_owned()
    }
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // HELPERS

    fn test(input: &str, expected: &str) {
        assert_eq!(
            rustdocify(input, "foo", None, Some("foo")),
            Ok(expected.to_owned())
        );
    }

    // ============================================================
    // MISC

    #[test]
    fn retain_exact_newlines() {
        let input = "## A\n## B\r\na\nb\r\n[x]: https://docs.rs/foo\n[y]: https://docs.rs/foo\r\n";
        let expected = "# A\n# B\r\na\nb\r\n[x]: crate\n[y]: crate\r\n";
        assert_eq!(rustdocify(input, "foo", None, None).unwrap(), expected);
    }

    // ============================================================
    // HEADERS - ERRORS

    #[test]
    fn non_first_top_level_header() {
        assert_eq!(
            rustdocify("## Foo\n# Bar", "foo", None, None),
            Err(Error::NonFirstTopLevelHeader("# Bar".to_owned()))
        );
    }

    // ============================================================
    // HEADERS - IGNORE

    #[test]
    fn ignore_headers_without_space_before() {
        test("#Foo\n##Bar", "#Foo\n##Bar");
    }

    // ============================================================
    // HEADERS

    #[test]
    fn convert_headers() {
        test(
            "# Foo\nfoo\n## Bar\nbar\n### Baz",
            "foo\n# Bar\nbar\n## Baz",
        );
    }

    // ============================================================
    // LINKS - ERRORS

    #[test]
    fn missing_version_in_url() {
        assert_eq!(
            rustdocify("[x]: https://docs.rs/foo", "foo", Some("0.1.0"), None),
            Err(Error::MissingVersionInUrl("https://docs.rs/foo".to_owned()))
        );
    }

    #[test]
    fn unrecognized_url_html() {
        let input = "[x]: https://docs.rs/foo/*/foo/hello_world.html";
        let expected = "https://docs.rs/foo/*/foo/hello_world.html".to_owned();
        assert_eq!(
            rustdocify(input, "foo", None, None),
            Err(Error::UnrecognizedUrl(expected))
        );
    }

    #[test]
    fn unrecognized_url_non_html() {
        let input = "[x]: https://docs.rs/foo/*/foo/hello_world.png";
        let expected = "https://docs.rs/foo/*/foo/hello_world.png".to_owned();
        assert_eq!(
            rustdocify(input, "foo", None, None),
            Err(Error::UnrecognizedUrl(expected))
        );
    }

    #[test]
    fn wrong_crate_name_in_url() {
        assert_eq!(
            rustdocify("[x]: https://docs.rs/foo/*/bar", "foo", None, Some("foo")),
            Err(Error::WrongCrateNameInUrl(
                "https://docs.rs/foo/*/bar".to_owned()
            ))
        );
    }

    #[test]
    fn wrong_version_in_url() {
        assert_eq!(
            rustdocify("[x]: https://docs.rs/foo/0.2.0", "foo", Some("0.1.0"), None),
            Err(Error::WrongVersionInUrl(
                "https://docs.rs/foo/0.2.0".to_owned()
            ))
        );
    }

    // ============================================================
    // LINKS - NOT LINK LINE

    #[test]
    fn no_close_bracket() {
        test("[x: https://docs.rs/foo", "[x: https://docs.rs/foo");
    }

    #[test]
    fn no_colon_after_close_bracket() {
        test("[x] https://docs.rs/foo", "[x] https://docs.rs/foo");
    }

    #[test]
    fn no_space_after_colon() {
        test("[x]:https://docs.rs/foo", "[x]:https://docs.rs/foo");
    }

    #[test]
    fn no_url() {
        test("[x]:   ", "[x]:   ");
    }

    // ============================================================
    // LINKS - IGNORE

    #[test]
    fn ignore_link_of_another_domain() {
        test(
            "[x]: https://example.com/foo",
            "[x]: https://example.com/foo",
        );
    }

    #[test]
    fn ignore_link_to_another_package() {
        test("[x]: https://docs.rs/bar", "[x]: https://docs.rs/bar");
    }

    #[test]
    fn ignore_link_to_another_package_which_has_this_package_as_prefix() {
        test("[x]: https://docs.rs/foobar", "[x]: https://docs.rs/foobar");
    }

    // ============================================================
    // LINKS - MISC

    #[test]
    fn retain_multiple_whitespace_before_url() {
        test("[x]:   https://docs.rs/foo", "[x]:   crate");
    }

    #[test]
    fn retain_stuff_after_url() {
        test("[x]: https://docs.rs/foo   hello", "[x]: crate   hello");
    }

    // ============================================================
    // LINKS - NO VERSION

    #[test]
    fn package() {
        test("[x]: https://docs.rs/foo", "[x]: crate");
    }

    #[test]
    fn package_fragment() {
        test("[x]: https://docs.rs/foo#fragment", "[x]: crate#fragment");
    }

    #[test]
    fn package_slash() {
        test("[x]: https://docs.rs/foo/", "[x]: crate");
    }

    #[test]
    fn package_slash_fragment() {
        test("[x]: https://docs.rs/foo/#fragment", "[x]: crate#fragment");
    }

    // ============================================================
    // LINKS - HAS VERSION, NO CRATE

    #[test]
    fn package_version() {
        test("[x]: https://docs.rs/foo/*", "[x]: crate");
    }

    #[test]
    fn package_version_fragment() {
        test("[x]: https://docs.rs/foo/*#fragment", "[x]: crate#fragment");
    }

    #[test]
    fn package_version_index() {
        test("[x]: https://docs.rs/foo/*/index.html", "[x]: crate");
    }

    #[test]
    fn package_version_index_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/index.html#fragment",
            "[x]: crate#fragment",
        );
    }

    #[test]
    fn package_version_slash() {
        test("[x]: https://docs.rs/foo/*/", "[x]: crate");
    }

    #[test]
    fn package_version_slash_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/#fragment",
            "[x]: crate#fragment",
        );
    }

    // ============================================================
    // LINKS - HAS CRATE, BUT NO MORE SEGMENTS

    #[test]
    fn package_version_crate() {
        test("[x]: https://docs.rs/foo/*/foo", "[x]: crate");
    }

    #[test]
    fn package_version_crate_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo#fragment",
            "[x]: crate#fragment",
        );
    }

    #[test]
    fn package_version_crate_index() {
        test("[x]: https://docs.rs/foo/*/foo/index.html", "[x]: crate");
    }

    #[test]
    fn package_version_crate_index_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/index.html#fragment",
            "[x]: crate#fragment",
        );
    }

    #[test]
    fn package_version_crate_slash() {
        test("[x]: https://docs.rs/foo/*/foo/", "[x]: crate");
    }

    #[test]
    fn package_version_crate_slash_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/#fragment",
            "[x]: crate#fragment",
        );
    }

    // ============================================================
    // LINKS - MODULE

    #[test]
    fn module() {
        test("[x]: https://docs.rs/foo/*/foo/a/b", "[x]: crate::a::b");
    }

    #[test]
    fn module_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b#fragment",
            "[x]: crate::a::b#fragment",
        );
    }

    #[test]
    fn module_index() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/index.html",
            "[x]: crate::a::b",
        );
    }

    #[test]
    fn module_index_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/index.html#fragment",
            "[x]: crate::a::b#fragment",
        );
    }

    #[test]
    fn module_slash() {
        test("[x]: https://docs.rs/foo/*/foo/a/b/", "[x]: crate::a::b");
    }

    #[test]
    fn module_slash_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/#fragment",
            "[x]: crate::a::b#fragment",
        );
    }

    // ============================================================
    // LINKS - ENUM

    #[test]
    fn root_enum() {
        test(
            "[x]: https://docs.rs/foo/*/foo/enum.Foo.html",
            "[x]: crate::Foo",
        );
    }

    #[test]
    fn root_enum_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/enum.Foo.html#fragment",
            "[x]: crate::Foo#fragment",
        );
    }

    #[test]
    fn root_enum_method() {
        test(
            "[x]: https://docs.rs/foo/*/foo/enum.Foo.html#method.bar",
            "[x]: crate::Foo::bar",
        );
    }

    #[test]
    fn root_enum_variant() {
        test(
            "[x]: https://docs.rs/foo/*/foo/enum.Foo.html#variant.Bar",
            "[x]: crate::Foo::Bar",
        );
    }

    #[test]
    fn module_enum() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/enum.Foo.html",
            "[x]: crate::a::b::Foo",
        );
    }

    #[test]
    fn module_enum_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/enum.Foo.html#fragment",
            "[x]: crate::a::b::Foo#fragment",
        );
    }

    #[test]
    fn module_enum_method() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/enum.Foo.html#method.bar",
            "[x]: crate::a::b::Foo::bar",
        );
    }

    #[test]
    fn module_enum_variant() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/enum.Foo.html#variant.Bar",
            "[x]: crate::a::b::Foo::Bar",
        );
    }

    // ============================================================
    // LINKS - FUNCTION

    #[test]
    fn root_function() {
        test(
            "[x]: https://docs.rs/foo/*/foo/fn.bar.html",
            "[x]: crate::bar",
        );
    }

    #[test]
    fn root_function_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/fn.bar.html#fragment",
            "[x]: crate::bar#fragment",
        );
    }

    #[test]
    fn module_function() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/fn.bar.html",
            "[x]: crate::a::b::bar",
        );
    }

    #[test]
    fn module_function_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/fn.bar.html#fragment",
            "[x]: crate::a::b::bar#fragment",
        );
    }

    // ============================================================
    // LINKS - MACRO

    #[test]
    fn root_macro() {
        test(
            "[x]: https://docs.rs/foo/*/foo/macro.bar.html",
            "[x]: crate::bar",
        );
    }

    #[test]
    fn root_macro_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/macro.bar.html#fragment",
            "[x]: crate::bar#fragment",
        );
    }

    #[test]
    fn module_macro() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/macro.bar.html",
            "[x]: crate::a::b::bar",
        );
    }

    #[test]
    fn module_macro_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/macro.bar.html#fragment",
            "[x]: crate::a::b::bar#fragment",
        );
    }

    // ============================================================
    // LINKS - STRUCT

    #[test]
    fn root_struct() {
        test(
            "[x]: https://docs.rs/foo/*/foo/struct.Foo.html",
            "[x]: crate::Foo",
        );
    }

    #[test]
    fn root_struct_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/struct.Foo.html#fragment",
            "[x]: crate::Foo#fragment",
        );
    }

    #[test]
    fn root_struct_method() {
        test(
            "[x]: https://docs.rs/foo/*/foo/struct.Foo.html#method.bar",
            "[x]: crate::Foo::bar",
        );
    }

    #[test]
    fn module_struct() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/struct.Foo.html",
            "[x]: crate::a::b::Foo",
        );
    }

    #[test]
    fn module_struct_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/struct.Foo.html#fragment",
            "[x]: crate::a::b::Foo#fragment",
        );
    }

    #[test]
    fn module_struct_method() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/struct.Foo.html#method.bar",
            "[x]: crate::a::b::Foo::bar",
        );
    }

    // ============================================================
    // LINKS - TRAIT

    #[test]
    fn root_trait() {
        test(
            "[x]: https://docs.rs/foo/*/foo/trait.Foo.html",
            "[x]: crate::Foo",
        );
    }

    #[test]
    fn root_trait_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/trait.Foo.html#fragment",
            "[x]: crate::Foo#fragment",
        );
    }

    #[test]
    fn root_trait_method() {
        test(
            "[x]: https://docs.rs/foo/*/foo/trait.Foo.html#tymethod.bar",
            "[x]: crate::Foo::bar",
        );
    }

    #[test]
    fn module_trait() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/trait.Foo.html",
            "[x]: crate::a::b::Foo",
        );
    }

    #[test]
    fn module_trait_fragment() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/trait.Foo.html#fragment",
            "[x]: crate::a::b::Foo#fragment",
        );
    }

    #[test]
    fn module_trait_method() {
        test(
            "[x]: https://docs.rs/foo/*/foo/a/b/trait.Foo.html#tymethod.bar",
            "[x]: crate::a::b::Foo::bar",
        );
    }

    // ============================================================
    // CODE BLOCKS

    #[test]
    fn two_backticks_doesnt_begin_code_block() {
        test("``\n## a", "``\n# a");
    }

    #[test]
    fn three_backtick_code_block_with_header() {
        test("```\n## a\n```\n## b", "```\n## a\n```\n# b");
    }

    #[test]
    fn four_backtick_code_block_with_header() {
        test("````\n```\n## a\n````\n## b", "````\n```\n## a\n````\n# b");
    }

    #[test]
    fn code_block_with_link() {
        test(
            "```\n[a]: https://docs.rs/foo\n```\n[b]: https://docs.rs/foo",
            "```\n[a]: https://docs.rs/foo\n```\n[b]: crate",
        );
    }

    #[test]
    fn code_block_with_short_line() {
        test("```\n\n```", "```\n\n```");
    }

    #[test]
    fn code_block_with_type() {
        test("```foo\n## a\n```\n## b", "```foo\n## a\n```\n# b");
    }
}
