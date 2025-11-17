create table "public"."games" (
    "id" uuid not null default gen_random_uuid(),
    "game_date" date not null,
    "home_team" text,
    "away_team" text
);


create table "public"."pitches" (
    "id" uuid not null default gen_random_uuid(),
    "game_id" uuid not null,
    "batter_id" uuid not null,
    "pitch_type" text not null,
    "speed" numeric(4,1),
    "description" text,
    "created_at" timestamp with time zone default now()
);


create table "public"."players" (
    "id" uuid not null default gen_random_uuid(),
    "first_name" text not null,
    "last_name" text not null,
    "position" text,
    "team" text
);


CREATE UNIQUE INDEX games_pkey ON public.games USING btree (id);

CREATE UNIQUE INDEX pitches_pkey ON public.pitches USING btree (id);

CREATE UNIQUE INDEX players_pkey ON public.players USING btree (id);

alter table "public"."games" add constraint "games_pkey" PRIMARY KEY using index "games_pkey";

alter table "public"."pitches" add constraint "pitches_pkey" PRIMARY KEY using index "pitches_pkey";

alter table "public"."players" add constraint "players_pkey" PRIMARY KEY using index "players_pkey";

alter table "public"."pitches" add constraint "pitches_batter_id_fkey" FOREIGN KEY (batter_id) REFERENCES players(id) ON DELETE CASCADE not valid;

alter table "public"."pitches" validate constraint "pitches_batter_id_fkey";

alter table "public"."pitches" add constraint "pitches_game_id_fkey" FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE not valid;

alter table "public"."pitches" validate constraint "pitches_game_id_fkey";



