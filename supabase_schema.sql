-- Run this once in the Supabase SQL editor (Dashboard -> SQL -> New query).
create extension if not exists "pgcrypto";

create table if not exists public.diagnoses (
    id                uuid primary key default gen_random_uuid(),
    created_at        timestamptz not null default now(),
    breed             text,
    breed_confidence  real,
    lesion            text,
    lesion_confidence real,
    lesion_category   text,
    low_confidence    boolean,
    treatment         jsonb,
    diet              jsonb,
    ai_recommendation text
);

create index if not exists diagnoses_created_at_idx
    on public.diagnoses (created_at desc);

-- The backend connects with the SERVICE ROLE key (server-side only), which bypasses
-- Row Level Security. Keep that key secret (Railway env var, never in the frontend).
