"""initial schema

Revision ID: 427c947baba1
Revises: 
Create Date: 2026-11-03 14:48:31.236015

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '427c947baba1'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('models',
    sa.Column('id', sa.String(length=120), nullable=False),
    sa.Column('model_type', sa.String(length=120), nullable=False),
    sa.Column('model_name', sa.String(length=4000), nullable=False),
    sa.Column('model_version', sa.String(length=4000), nullable=False),
    sa.Column('artifact_path', sa.String(length=4000), nullable=True),
    sa.Column('feature_names', sa.JSON(), nullable=True),
    sa.Column('mlflow_run_id', sa.String(length=120), nullable=True),
    sa.Column('shap_artifact_path', sa.String(length=4000), nullable=True),
    sa.Column('shap_logged_at', sa.DateTime(), nullable=True),
    sa.Column('embedding_artifact_path', sa.String(length=4000), nullable=True),
    sa.Column('embedding_logged_at', sa.DateTime(), nullable=True),
    sa.Column('created', sa.DateTime(), nullable=True),
    sa.Column('updated', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_models'))
    )
    op.create_table('predictions',
    sa.Column('id', sa.String(length=120), nullable=False),
    sa.Column('features', sa.JSON(), nullable=False),
    sa.Column('tags', sa.JSON(), nullable=False),
    sa.Column('label', sa.JSON(), nullable=False),
    sa.Column('actual', sa.JSON(), nullable=True),
    sa.Column('threshold', sa.Float(), nullable=True),
    sa.Column('embeddings', sa.JSON(), nullable=True),
    sa.Column('created', sa.DateTime(), nullable=True),
    sa.Column('updated', sa.DateTime(), nullable=True),
    sa.Column('model_id', sa.String(length=120), nullable=False),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], name=op.f('fk_predictions_model_id_models')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_predictions'))
    )
    op.create_table('explanations',
    sa.Column('id', sa.String(length=120), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('values', sa.JSON(), nullable=False),
    sa.Column('prediction_id', sa.String(length=120), nullable=False),
    sa.ForeignKeyConstraint(['prediction_id'], ['predictions.id'], name=op.f('fk_explanations_prediction_id_predictions')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_explanations'))
    )


def downgrade():
    op.drop_table('explanations')
    op.drop_table('predictions')
    op.drop_table('models')
