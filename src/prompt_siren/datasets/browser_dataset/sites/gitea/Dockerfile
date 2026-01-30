# Pre-seeded Gitea image for browser dataset
#
# This image contains a pre-populated SQLite database with test data
# including injection vector placeholders like {gitea_readme_content}.
#
# The database is regenerated via the seeder callback defined in the site
# configuration. The seeder runs before each image build when needed.

FROM gitea/gitea:latest

# Run as root so Gitea can bind to privileged port 80, matching benchmark URLs.
ENV USER_UID=0 \
    USER_GID=0 \
    GITEA__server__HTTP_PORT=80 \
    GITEA__server__ROOT_URL=http://gitea.dev-forge.io

EXPOSE 80

# Copy pre-seeded database (contains repos, issues, and comments)
COPY data/gitea.db /data/gitea/gitea.db

# Ensure correct permissions
RUN chown -R git:git /data/gitea
