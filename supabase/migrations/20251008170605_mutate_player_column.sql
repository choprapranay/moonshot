ALTER TABLE "public"."players"
DROP COLUMN "position",
DROP COLUMN "team",
DROP COLUMN "first_name",
DROP COLUMN "last_name",
DROP COLUMN "name",
ADD COLUMN "name" text not null,
ADD COLUMN "created_at" timestamp default now();