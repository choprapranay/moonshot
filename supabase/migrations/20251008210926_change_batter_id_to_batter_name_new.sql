ALTER TABLE "public"."pitches" DROP COLUMN "batter_name";
ALTER TABLE "public"."pitches" ADD COLUMN "batter_name" text;
DROP TABLE "public"."players";