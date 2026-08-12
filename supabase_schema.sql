-- Run this once in the Supabase SQL editor (Dashboard -> SQL -> New query).
create extension if not exists "pgcrypto";

create table if not exists public.diagnoses (
    id                uuid primary key default gen_random_uuid(),
    created_at        timestamptz not null default now(),
    user_email        text,
    breed             text,
    breed_confidence  real,
    lesion            text,
    lesion_confidence real,
    lesion_category   text,
    low_confidence    boolean,
    treatment         jsonb,
    diet              jsonb,
    ai_recommendation text,
    image_url         text,
    lesion_image_url  text
);

-- If the table already exists from an earlier version, add the new columns:
alter table public.diagnoses add column if not exists user_email       text;
alter table public.diagnoses add column if not exists image_url        text;
alter table public.diagnoses add column if not exists lesion_image_url text;

create index if not exists diagnoses_created_at_idx on public.diagnoses (created_at desc);
create index if not exists diagnoses_user_email_idx on public.diagnoses (user_email);

-- Storage: create a PUBLIC bucket named 'images' (Dashboard -> Storage -> New bucket -> Public)
-- so upload_image() can store the dog/lesion photos and return public URLs.

-- The backend connects with the SERVICE ROLE key (server-side only), which bypasses
-- Row Level Security. Keep that key secret (Railway env var, never in the frontend / never in git).
