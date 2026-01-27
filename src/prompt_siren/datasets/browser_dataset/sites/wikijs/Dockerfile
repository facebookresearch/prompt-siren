# Pre-seeded Wiki.js image for browser dataset
#
# This image contains a pre-populated SQLite database with documentation pages
# including injection vector placeholders like {wikijs_home_content}.
#
# The database is regenerated via the seeder callback defined in the site
# configuration. The seeder runs before each image build when needed.

FROM linuxserver/wikijs:latest

# Copy pre-seeded database (contains documentation pages)
COPY data/database.sqlite /config/database.sqlite

# Ensure correct permissions
RUN chown -R abc:abc /config
