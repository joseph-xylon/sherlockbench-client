"""add failure_info to runs

Revision ID: add_failure_info_to_runs
Revises: 9ba4029805cf
Create Date: 2025-03-05 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = 'add_failure_info_to_runs'
down_revision: Union[str, None] = '9ba4029805cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the failure_info column to the runs table
    op.add_column('runs', sa.Column('failure_info', JSONB, nullable=True))


def downgrade() -> None:
    # Drop the failure_info column from the runs table
    op.drop_column('runs', 'failure_info')