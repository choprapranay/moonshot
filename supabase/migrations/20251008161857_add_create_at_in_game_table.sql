alter table "public"."games"
add column "created_at" timestamp with time zone default now();
