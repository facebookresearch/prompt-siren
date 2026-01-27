# Pre-seeded Apache Answer image for browser dataset
#
# This image contains a pre-populated SQLite database with test data
# including injection vector placeholders like {answer_question_body}.
#
# The database is regenerated via the seeder callback defined in the site
# configuration. The seeder runs before each image build when needed.

FROM apache/answer:latest

# Copy pre-seeded database (contains questions, answers, comments)
COPY data/answer.db /data/answer.db

# Ensure correct permissions
RUN chown -R answer:answer /data
