# Pre-seeded Wiki.js image for browser dataset
#
# This image contains a pre-populated SQLite database with documentation pages
# including injection vector placeholders like {wikijs_home_content}.
#
# The database is regenerated via the seeder callback defined in the site
# configuration. The seeder runs before each image build when needed.

FROM linuxserver/wikijs:latest

# Run as root so Wiki.js can bind to privileged port 80, matching benchmark URLs.
ENV PUID=0 \
    PGID=0 \
    PORT=80 \
    WIKIJS_PORT=80

EXPOSE 80

# Copy pre-seeded database (contains documentation pages)
COPY data/db.sqlite /config/db.sqlite

# Configure Wiki.js to listen on port 80
COPY config.yml /config/config.yml

# Ensure correct permissions
RUN chown -R abc:abc /config
