# Contributing to Hezar
Welcome to Hezar! We greatly appreciate your interest in contributing to this project and helping us make it even more
valuable to the Persian community. Whether you're a developer, researcher, or enthusiast, your contributions are
invaluable in helping us grow and improve Hezar.

Before you start contributing, please take a moment to review the following guidelines.

## Code of Conduct

This project and its community adhere to
the [Contributor Code of Conduct](https://github.com/hezarai/hezar/blob/main/CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

If you come across a bug or unexpected behavior, please help us by reporting it.
Use the [GitHub Issue Tracker](https://github.com/hezarai/hezar/issues) to create a detailed bug report.
Include information such as:

- A clear and descriptive title.
- Steps to reproduce the bug.
- Expected behavior.
- Actual behavior.
- Your operating system and Python version.

### Adding features

Have a great idea for a new feature or improvement? We'd love to hear it. You can open an issue and add your suggestion
with a clear description and further suggestions on how it can be implemented. Also, if you already can implement it
yourself, just follow the instructions on how you can send a PR.

### Adding/Improving documents

Have a suggestion to enhance our documentation or want to contribute entirely new sections? We welcome your input!<br>
Here's how you can get involved:<br>
Docs website is deployed here: [https://hezarai.github.io/hezar](https://hezarai.github.io/hezar) and the source for the
docs are located at the [docs](https://github.com/hezarai/hezar/tree/main/docs) folder in the root of the repo. Feel
free to apply your changes or add new docs to this section. Notice that docs are written in Markdown format. In case you have
added new files to this section, you must include them in the `index.md` file in the same folder. For example, if you've
added the file `new_doc.md` to the `get_started` folder, you have to modify `get_started/index.md` and put your file
name there.

### Commit guidelines

#### Functional best practices

- Ensure only one "logical change" per commit for efficient review and flaw identification.
- Smaller code changes facilitate quicker reviews and easier troubleshooting using Git's bisect capability.
- Avoid mixing whitespace changes with functional code changes.
- Avoid mixing two unrelated functional changes.
- Refrain from sending large new features in a single giant commit.

#### Styling best practices
- Use imperative mood in the subject (e.g., "Add support for ..." not "Adding support or added support") .
- Keep the subject line short and concise, preferably less than 50 characters.
- Capitalize the subject line and do not end it with a period.
- Wrap body lines at 72 characters.
- Use the body to explain what and why a change was made.
- Do not explain the "how" in the commit message; reserve it for documentation or code.
- For commits referencing an issue or pull request, write the proper commit subject followed by the reference in parentheses (e.g., "Add NFKC normalizer (#9999)").
- Reference codes & paths in back quotes (e.g., `variable`, `method()`, `Class()`, `file.py`).
- Preferably use the following [gitmoji](https://gitmoji.dev/) compatible codes at the beginning of your commit message:
  - `:bug:`: Fix a bug or issue
  - `:sparkles`: Add feature or improvements
  - `:recycle:`: Refactor code (backward compatible refactor)
  - `:memo:`: Add or change docs
  - `:pencil2:`: Minor change or improvement
  - `:fire:`: Remove code or file
  - `:boom:`: Introduce breaking or backward-incompatible changes
  - `:bookmark:`: Version release
  - `:adhesive_bandage:`: Non-critical fix

## Sending a PR

In order to apply any change to the repo, you have to follow these step:

1. Fork the Hezar repository.
2. Create a new branch for your feature, bug fix, etc.
3. Make your changes.
4. Update the documentation to reflect your changes.
5. Ensure your code adheres to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
6. Format the code using `ruff` (`ruff check --fix .`)
7. Write tests to ensure the functionality if needed.
8. Run tests and make sure all of them pass. (Skip this step if your changes do not involve codes)
9. Open a pull request from your fork and the PR template will be automatically loaded to help you do the rest.
10. Be responsive to feedback and comments during the review process.
11. Thanks for contributing to the Hezar project.😉❤️

## License

By contributing to Hezar, you agree that your contributions will be licensed under
the [Apache 2.0 License](https://github.com/hezarai/hezar/blob/main/LICENSE).

We look forward to your contributions and appreciate your efforts in making Hezar a powerful AI tool for the Persian
community!