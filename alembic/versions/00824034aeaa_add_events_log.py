"""add events log

Revision ID: 00824034aeaa
Revises: ee7a4f17393e
Create Date: 2025-07-19 05:55:59.636832

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = '00824034aeaa'
down_revision: Union[str, None] = 'ee7a4f17393e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "event_counts",
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column('event_name', sa.String, nullable=False),
        sa.Column('count', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('run_id', 'event_name')
    )

def downgrade() -> None:
    op.drop_table('event_counts')
