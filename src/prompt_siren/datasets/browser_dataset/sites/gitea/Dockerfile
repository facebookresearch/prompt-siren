# Pre-seeded Gitea image for browser dataset
#
# This image contains a pre-populated SQLite database with test data
# including injection vector placeholders like {gitea_readme_content}.
#
# The database is regenerated via the seeder callback defined in the site
# configuration. The seeder runs before each image build when needed.

FROM gitea/gitea:latest

# Copy pre-seeded database (contains repos, issues, and comments)
COPY data/gitea.db /data/gitea/gitea.db

# Ensure correct permissions
RUN chown -R git:git /data/gitea
