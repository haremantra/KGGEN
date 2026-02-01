-- KGGEN-CUAD PostgreSQL Schema Initialization
-- This script runs automatically when the PostgreSQL container starts

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity searches

-- =============================================================================
-- CONTRACTS TABLE
-- Stores contract metadata and processing status
-- =============================================================================
CREATE TABLE contracts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cuad_id VARCHAR(100) UNIQUE NOT NULL,
    filename VARCHAR(500) NOT NULL,
    contract_type VARCHAR(100),
    jurisdiction VARCHAR(100),
    raw_text TEXT,
    page_count INTEGER,
    word_count INTEGER,
    status VARCHAR(50) DEFAULT 'pending',  -- pending, extracting, extracted, aggregating, aggregated, resolving, resolved, failed
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    extracted_at TIMESTAMP WITH TIME ZONE,
    aggregated_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- =============================================================================
-- EXTRACTIONS TABLE
-- Stores raw extraction outputs from Stage 1
-- =============================================================================
CREATE TABLE extractions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    extraction_type VARCHAR(50) NOT NULL,  -- 'entity' or 'relation'
    data JSONB NOT NULL,
    llm_model VARCHAR(100),
    llm_provider VARCHAR(50),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    confidence_score FLOAT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- ENTITIES TABLE
-- Stores extracted entities with their types and properties
-- =============================================================================
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,  -- Party, IPAsset, Obligation, Restriction, LiabilityProvision, Temporal, Jurisdiction, ContractClause
    properties JSONB DEFAULT '{}',
    source_text TEXT,
    source_page INTEGER,
    confidence_score FLOAT,
    is_canonical BOOLEAN DEFAULT FALSE,
    canonical_id UUID REFERENCES entities(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- TRIPLES TABLE
-- Stores knowledge graph triples (subject-predicate-object)
-- =============================================================================
CREATE TABLE triples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    subject_id UUID REFERENCES entities(id),
    subject_text VARCHAR(500) NOT NULL,
    predicate VARCHAR(200) NOT NULL,  -- LICENSES_TO, OWNS, ASSIGNS, HAS_OBLIGATION, etc.
    object_id UUID REFERENCES entities(id),
    object_text VARCHAR(500) NOT NULL,
    properties JSONB DEFAULT '{}',
    cuad_label VARCHAR(100),
    confidence_score FLOAT,
    source_text TEXT,
    source_page INTEGER,
    is_canonical BOOLEAN DEFAULT FALSE,
    canonical_id UUID REFERENCES triples(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- ENTITY_ALIASES TABLE
-- Maps variant forms to canonical entities (from Stage 3 resolution)
-- =============================================================================
CREATE TABLE entity_aliases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    canonical_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    canonical_form VARCHAR(500) NOT NULL,
    alias VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100),
    similarity_score FLOAT,
    resolution_method VARCHAR(50),  -- 'exact', 'semantic', 'llm'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(canonical_form, alias)
);

-- =============================================================================
-- PREDICATE_ALIASES TABLE
-- Maps variant predicates to canonical forms
-- =============================================================================
CREATE TABLE predicate_aliases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    canonical_form VARCHAR(200) NOT NULL,
    alias VARCHAR(200) NOT NULL,
    similarity_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(canonical_form, alias)
);

-- =============================================================================
-- EMBEDDINGS TABLE
-- Stores vector embeddings for entities and triples
-- =============================================================================
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    triple_id UUID REFERENCES triples(id) ON DELETE CASCADE,
    embedding_type VARCHAR(50) NOT NULL,  -- 'entity', 'triple', 'query'
    model_name VARCHAR(100) NOT NULL,
    vector BYTEA NOT NULL,  -- Stored as binary, use Qdrant for actual search
    dimension INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_entity_or_triple CHECK (
        (entity_id IS NOT NULL AND triple_id IS NULL) OR
        (entity_id IS NULL AND triple_id IS NOT NULL)
    )
);

-- =============================================================================
-- PROCESSING_JOBS TABLE
-- Tracks async processing jobs
-- =============================================================================
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL,  -- 'extraction', 'aggregation', 'resolution', 'query'
    status VARCHAR(50) DEFAULT 'pending',  -- pending, running, completed, failed
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    progress FLOAT DEFAULT 0.0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- QUERY_CACHE TABLE
-- Caches query results for performance
-- =============================================================================
CREATE TABLE query_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    contract_ids UUID[],
    result JSONB NOT NULL,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Contracts indexes
CREATE INDEX idx_contracts_status ON contracts(status);
CREATE INDEX idx_contracts_type ON contracts(contract_type);
CREATE INDEX idx_contracts_cuad_id ON contracts(cuad_id);
CREATE INDEX idx_contracts_created_at ON contracts(created_at);

-- Entities indexes
CREATE INDEX idx_entities_contract ON entities(contract_id);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_name_trgm ON entities USING gin(name gin_trgm_ops);
CREATE INDEX idx_entities_canonical ON entities(canonical_id) WHERE canonical_id IS NOT NULL;

-- Triples indexes
CREATE INDEX idx_triples_contract ON triples(contract_id);
CREATE INDEX idx_triples_subject ON triples(subject_text);
CREATE INDEX idx_triples_predicate ON triples(predicate);
CREATE INDEX idx_triples_object ON triples(object_text);
CREATE INDEX idx_triples_cuad_label ON triples(cuad_label);
CREATE INDEX idx_triples_subject_trgm ON triples USING gin(subject_text gin_trgm_ops);
CREATE INDEX idx_triples_object_trgm ON triples USING gin(object_text gin_trgm_ops);

-- Extractions indexes
CREATE INDEX idx_extractions_contract ON extractions(contract_id);
CREATE INDEX idx_extractions_type ON extractions(extraction_type);

-- Entity aliases indexes
CREATE INDEX idx_entity_aliases_canonical ON entity_aliases(canonical_id);
CREATE INDEX idx_entity_aliases_alias ON entity_aliases(alias);

-- Processing jobs indexes
CREATE INDEX idx_jobs_status ON processing_jobs(status);
CREATE INDEX idx_jobs_type ON processing_jobs(job_type);

-- Query cache indexes
CREATE INDEX idx_query_cache_hash ON query_cache(query_hash);
CREATE INDEX idx_query_cache_expires ON query_cache(expires_at);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for contracts table
CREATE TRIGGER update_contracts_updated_at
    BEFORE UPDATE ON contracts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Insert predicate types (canonical forms)
INSERT INTO predicate_aliases (canonical_form, alias, similarity_score) VALUES
    ('LICENSES_TO', 'licenses_to', 1.0),
    ('LICENSES_TO', 'grants_license_to', 0.95),
    ('LICENSES_TO', 'licenses', 0.9),
    ('OWNS', 'owns', 1.0),
    ('OWNS', 'has_ownership_of', 0.95),
    ('ASSIGNS', 'assigns', 1.0),
    ('ASSIGNS', 'transfers', 0.9),
    ('ASSIGNS', 'assigns_to', 0.95),
    ('HAS_OBLIGATION', 'has_obligation', 1.0),
    ('HAS_OBLIGATION', 'must', 0.85),
    ('HAS_OBLIGATION', 'shall', 0.85),
    ('SUBJECT_TO_RESTRICTION', 'subject_to_restriction', 1.0),
    ('SUBJECT_TO_RESTRICTION', 'restricted_by', 0.9),
    ('SUBJECT_TO_RESTRICTION', 'cannot', 0.85),
    ('HAS_LIABILITY', 'has_liability', 1.0),
    ('HAS_LIABILITY', 'is_liable_for', 0.95),
    ('GOVERNED_BY', 'governed_by', 1.0),
    ('GOVERNED_BY', 'subject_to_law_of', 0.9),
    ('CONTAINS_CLAUSE', 'contains_clause', 1.0),
    ('CONTAINS_CLAUSE', 'includes_clause', 0.95),
    ('EFFECTIVE_ON', 'effective_on', 1.0),
    ('EFFECTIVE_ON', 'commences_on', 0.9),
    ('TERMINATES_ON', 'terminates_on', 1.0),
    ('TERMINATES_ON', 'expires_on', 0.95),
    ('TERMINATES_ON', 'ends_on', 0.9);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO kggen;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO kggen;
