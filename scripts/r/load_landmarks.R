# Landmarks exported from caliPr, and the standard analyses on them.
# Study: @STUDY@
#
# Run it from this folder:   Rscript load_landmarks.R
# It reads the four files beside it and writes shape_space.pdf.

library(geomorph)

# ---- read ------------------------------------------------------------------
# negNA = TRUE turns the -1 placeholders into NA for landmarks nobody placed.
# Without it they become real points at the image corner and drag the fit.
#
# When EVERY specimen carries a scale, the .tps has a SCALE line for each and
# readland.tps applies it, so the coordinates arrive in millimetres. When only
# some do, the .tps is left in pixels throughout -- rescaling some and not others
# would make centroid size mean two different things in one series -- and
# specimens.csv carries each fish's own px/mm to put size in millimetres below.
A  <- readland.tps("landmarks.tps", specID = "ID", negNA = TRUE, warnmsg = FALSE)
nm <- read.csv("landmark_names.csv", stringsAsFactors = FALSE)
dimnames(A)[[1]] <- nm$name
sp <- read.csv("specimens.csv", stringsAsFactors = FALSE)
sp <- sp[match(dimnames(A)[[3]], sp$ID), ]
stopifnot(identical(sp$ID, dimnames(A)[[3]]))

# ---- landmark set ----------------------------------------------------------
# Fin tips sit wherever the fin happened to dry: folded, splayed, pinned. On a
# preserved series that is posture, not shape, and it shows up as a component of
# its own. Set to FALSE to keep them.
drop_fin_tips <- TRUE
tips <- c("dorsal_tip", "pelvic_tip", "anal_tip", "pectoral_ray_tip")
if (drop_fin_tips) A <- A[!(dimnames(A)[[1]] %in% tips), , , drop = FALSE]

# ---- missing landmarks -----------------------------------------------------
# gpagen() will not run with NA present. The default keeps complete specimens.
# The alternative, estimate.missing(), interpolates a missing point from the
# rest of the fish -- defensible for a few, but here the gaps sit in the fins
# that were hardest to see, so the interpolated points would be least
# trustworthy exactly where they would be read.
complete <- apply(A, 3, function(m) !anyNA(m))
cat(sum(!complete), "of", length(complete),
    "specimens have a missing landmark and are left out\n")
A  <- A[, , complete]
sp <- sp[complete, ]
# A <- estimate.missing(A, method = "TPS")   # instead of the three lines above

group <- factor(sp$group)
print(table(group))

# ---- Procrustes superimposition -------------------------------------------
gpa <- gpagen(A, print.progress = FALSE)
dimnames(gpa$coords) <- dimnames(A)      # gpagen drops the landmark names

# Centroid size in millimetres. If the .tps carried SCALE lines it already is;
# dividing by px/mm again made it 20-25x too small, by a different factor for
# every fish. Otherwise each fish's own scale converts it, and a fish with no
# scale uses the series' median px/mm: the rig's magnification varies by about
# 1.5% between photographs, far less than size varies between fish.
scaled <- any(grepl("^SCALE=", readLines("landmarks.tps")))
ppm  <- ifelse(is.na(sp$px_per_mm), median(sp$px_per_mm, na.rm = TRUE), sp$px_per_mm)
size <- if (scaled) gpa$Csize else gpa$Csize / ppm

# ---- body arching ----------------------------------------------------------
# Preserved fish lie curved, and how curved is decided on the tray, not by the
# fish. It is typically the largest single source of variation in a series like
# this one: the midbody sits above or below the line from snout to caudal base.
# Measured here as that offset, as a fraction of the line's length.
arch <- apply(gpa$coords, 3, function(m) {
  a <- m["premaxilla_tip", ]; b <- m["caudal_base", ]
  mid <- (m["dorsal_base_center", ] + m["pelvic_base_center", ]) / 2
  u <- (b - a) / sqrt(sum((b - a)^2))
  ((mid[1] - a[1]) * (-u[2]) + (mid[2] - a[2]) * u[1]) / sqrt(sum((b - a)^2))
})

gdf <- geomorph.data.frame(coords = gpa$coords, group = group,
                           logsize = log(size), arch = arch)

# ---- is the arching confounded with group? --------------------------------
cat("\nArching by group -- if this differs, an uncorrected test of group\n")
cat("is partly a test of how the fish were laid on the tray:\n")
print(round(tapply(arch, group, mean), 4))
print(anova(lm(arch ~ group)))

# ---- shape space, raw and with the arching removed -------------------------
pca_raw <- gm.prcomp(gpa$coords)
cat(sprintf("\nraw PCA: PC1 %.1f%% of shape variance, correlation with arching r = %.2f\n",
            100 * pca_raw$d[1] / sum(pca_raw$d), cor(pca_raw$x[, 1], arch)))

Y  <- two.d.array(gpa$coords)
Yu <- resid(lm(Y ~ arch)) + matrix(colMeans(Y), nrow(Y), ncol(Y), byrow = TRUE)
unbent <- arrayspecs(Yu, dim(gpa$coords)[1], 2)
dimnames(unbent) <- dimnames(gpa$coords)
pca <- gm.prcomp(unbent)
summary(pca)

# ---- tests -----------------------------------------------------------------
# Sequential sums of squares: group is tested after arching and size are taken
# out, which is the question worth asking -- do the groups differ in shape, over
# and above how the fish were laid out and how big they are?
fit_null <- procD.lm(coords ~ arch + logsize, data = gdf, iter = 999, SS.type = "I",
                     print.progress = FALSE)
fit      <- procD.lm(coords ~ arch + logsize + group, data = gdf, iter = 999,
                     SS.type = "I", print.progress = FALSE)
cat("\n==== shape ~ arching + size + group ====\n")
print(summary(fit))

# Do the groups share an allometry? If not, "the groups differ in shape" has to
# be read at a stated size.
fit_slopes <- procD.lm(coords ~ arch + logsize * group, data = gdf, iter = 999,
                       SS.type = "I", print.progress = FALSE)
cat("\n==== do the groups share a size-shape slope? ====\n")
print(anova(fit, fit_slopes, print.progress = FALSE))

# Which groups differ from which.
cat("\n==== pairwise, after arching and size ====\n")
pw <- pairwise(fit, fit_null, groups = group)
print(summary(pw, test.type = "dist", confidence = 0.95))

# For comparison only: the test as it would run without correcting anything.
cat("\n==== for comparison: shape ~ group with nothing taken out ====\n")
print(summary(procD.lm(coords ~ group, data = gdf, iter = 999,
                       print.progress = FALSE)))

# ---- figure ----------------------------------------------------------------
pdf("shape_space.pdf", width = 7, height = 6)
cols <- c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")[seq_along(levels(group))]
plot(pca, pch = 19, col = cols[group], cex = 1.1,
     main = "@STUDY@ shape space, arching removed")
legend("topright", legend = levels(group), col = cols, pch = 19, bty = "n")
dev.off()
cat("\nwrote shape_space.pdf\n")

# ---------------------------------------------------------------------------
# landmarks_imagej.csv holds the same points in the shape ImageJ's Multi-Measure
# writes, for pooling with a series digitised there. Mind the y axis: that file
# measures down from the top of the photograph, the .tps above measures up, so
# the two are mirror images -- pick one and stay with it. Its units column says
# mm or px per row.
#
# lm <- read.csv("landmarks_imagej.csv", stringsAsFactors = FALSE)
# k  <- length(unique(lm$landmark))
# B  <- arrayspecs(as.matrix(lm[, c("X", "Y")]), p = k, k = 2)
# dimnames(B)[[1]] <- unique(lm$landmark)
# dimnames(B)[[3]] <- unique(lm$Label)
