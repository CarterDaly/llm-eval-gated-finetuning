"""llm models and generations

Revision ID: dd671692612a
Revises: 427c947baba1
Create Date: 2026-11-04 12:58:33.310460

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'dd671692612a'
down_revision = '427c947baba1'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('generations',
    sa.Column('id', sa.String(length=120), nullable=False),
    sa.Column('model_id', sa.String(length=120), nullable=False),
    sa.Column('messages', sa.JSON(), nullable=False),
    sa.Column('tools', sa.JSON(), nullable=True),
    sa.Column('params', sa.JSON(), nullable=False),
    sa.Column('output_text', sa.Text(), nullable=True),
    sa.Column('finish_reason', sa.String(length=40), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('backend', sa.String(length=40), nullable=False),
    sa.Column('served_model', sa.String(length=400), nullable=True),
    sa.Column('prompt_tokens', sa.Integer(), nullable=True),
    sa.Column('completion_tokens', sa.Integer(), nullable=True),
    sa.Column('latency_ms', sa.Float(), nullable=True),
    sa.Column('cost_per_1k_tokens', sa.Float(), nullable=True),
    sa.Column('cost_usd', sa.Float(), nullable=True),
    sa.Column('prompt_embedding', sa.JSON(), nullable=True),
    sa.Column('embedding_model', sa.String(length=400), nullable=True),
    sa.Column('source', sa.String(length=20), server_default='live', nullable=False),
    sa.Column('example_id', sa.String(length=400), nullable=True),
    sa.Column('tags', sa.JSON(), nullable=False),
    sa.Column('created', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], name=op.f('fk_generations_model_id_models')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_generations'))
    )
    with op.batch_alter_table('generations', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_generations_example_id'), ['example_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_generations_model_id'), ['model_id'], unique=False)

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('kind', sa.String(length=20), server_default='classical', nullable=False))
        batch_op.add_column(sa.Column('base_model', sa.String(length=400), nullable=True))
        batch_op.add_column(sa.Column('base_model_revision', sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column('adapter_uri', sa.String(length=4000), nullable=True))
        batch_op.add_column(sa.Column('adapter_sha256', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('generation_config', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('parent_model_id', sa.String(length=120), nullable=True))
        batch_op.create_foreign_key(batch_op.f('fk_models_parent_model_id_models'), 'models', ['parent_model_id'], ['id'])


def downgrade():
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f('fk_models_parent_model_id_models'), type_='foreignkey')
        batch_op.drop_column('parent_model_id')
        batch_op.drop_column('generation_config')
        batch_op.drop_column('adapter_sha256')
        batch_op.drop_column('adapter_uri')
        batch_op.drop_column('base_model_revision')
        batch_op.drop_column('base_model')
        batch_op.drop_column('kind')

    with op.batch_alter_table('generations', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_generations_model_id'))
        batch_op.drop_index(batch_op.f('ix_generations_example_id'))

    op.drop_table('generations')
